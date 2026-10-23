"""VLM 2D grounding as a detector: Cosmos 3 (Nano/Edge) and Qwen3-VL on a fixed KITTI subset.

The model gets one prompt per frame listing the target categories and must answer with JSON boxes.
Qwen3-VL (which Cosmos 3 Nano's reasoner is initialized from) answers like
    [{"bbox_2d": [x1, y1, x2, y2], "label": "car"}, ...]   with coordinates on a 0-1000 scale.
Neither Cosmos 3 model card documents its box format, so the smoke test draws every parsed box on
the image; confirm the convention there before running the full subset.

VLMs give no per-box confidence, so every detection is written with score 1.0 and the evaluator
reports operating points (precision/recall/F1 at IoU 0.5 and 0.7) instead of AP.

Backends:
    transformers   in-process, BF16 (Cosmos3OmniForConditionalGeneration for Nano,
                   AutoModelForImageTextToText for Edge and Qwen3-VL)
    openai         any OpenAI-compatible server, e.g. `vllm serve nvidia/Cosmos3-Nano ...`

Usage:
    python -m det_eval.vlm_grounding --model cosmos3_nano --limit 20 --draw runs/vlm_smoke/cosmos3_nano
    python -m det_eval.vlm_grounding --model cosmos3_nano --backend openai --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from .common import load_json, load_yaml, read_ids, save_json, write_manifest

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
FENCE_RE = re.compile(r"```(?:json|python|JSON)?\s*(.*?)```", re.S)
OBJECT_RE = re.compile(r"\{[^{}]*\}")
BOX_KEYS = ("bbox_2d", "bbox", "box", "box_2d", "bbox2d", "bounding_box")
LABEL_KEYS = ("label", "category", "class", "name", "object")


# --------------------------------------------------------------------------- parsing

def _is_box(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v)


def _loads(s: str):
    for fn in (json.loads, ast.literal_eval):
        try:
            obj = fn(s)
        except Exception:
            continue
        # literal_eval turns "{...}, {...}" into a tuple; that is a fragment, not a complete answer,
        # so leave it to the per-object salvage path (which reports status "salvaged")
        if not isinstance(obj, tuple):
            return obj
    return None


def _strip_reasoning(text: str) -> str:
    text = THINK_RE.sub("", text)
    if "</think>" in text:                                  # opening tag was in the prompt template
        text = text.split("</think>", 1)[1]
    return text.strip()


def _candidates(text: str):
    for m in FENCE_RE.finditer(text):
        yield m.group(1).strip()
    i, j = text.find("["), text.rfind("]")
    if i != -1 and j > i:
        yield text[i:j + 1]
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        yield text[i:j + 1]


def _items(obj) -> list[dict]:
    """Normalize the shapes models produce into a list of {box-key, label-key} dicts."""
    if isinstance(obj, dict):
        if any(k in obj for k in BOX_KEYS):
            return [obj]
        items = []
        for key, val in obj.items():                        # {"objects": [...]} or {"car": [[...], ...]}
            if isinstance(val, list):
                for e in val:
                    if isinstance(e, dict):
                        items.append(e)
                    elif _is_box(e):
                        items.append({"bbox_2d": e, "label": key})
        return items
    if isinstance(obj, list):
        out = []
        for e in obj:
            if isinstance(e, dict):
                out.append(e)
            elif _is_box(e):
                out.append({"bbox_2d": e, "label": None})
        return out
    return []


def to_pixels(box, width: int, height: int, coord_mode: str = "norm1000") -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    if coord_mode == "auto":
        coord_mode = "norm1" if max(x1, y1, x2, y2) <= 1.0 else "norm1000"
    if coord_mode == "norm1000":
        sx, sy = width / 1000.0, height / 1000.0
    elif coord_mode == "norm1":
        sx, sy = float(width), float(height)
    elif coord_mode == "pixel":
        sx = sy = 1.0
    else:
        raise ValueError(f"unknown coord_mode {coord_mode!r}")
    x1, x2 = sorted((x1 * sx, x2 * sx))
    y1, y2 = sorted((y1 * sy, y2 * sy))
    return [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]


COMPACT_RE = re.compile(r"^[ \t]*([A-Za-z][A-Za-z _-]*?)[ \t]+(-?\d+(?:\.\d+)?)[ \t]+(-?\d+(?:\.\d+)?)"
                        r"[ \t]+(-?\d+(?:\.\d+)?)[ \t]+(-?\d+(?:\.\d+)?)[ \t]*$", re.M)


def parse_compact(text: str, width: int, height: int, coord_mode: str = "norm1000") -> tuple[list[dict], str] | None:
    """The compact format the speed-tuned adapter emits: one `label x1 y1 x2 y2` per line, `none` if empty.
    Returns None when the text is not in this format. Each box carries its character span for scoring."""
    if text.strip().lower() in ("none", "none."):
        return [], "empty"
    boxes = []
    for m in COMPACT_RE.finditer(text):
        x1, y1, x2, y2 = to_pixels([float(m.group(i)) for i in range(2, 6)], width, height, coord_mode)
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            boxes.append({"label": m.group(1).strip(), "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                          "span": [m.start(), m.end()]})
    return (boxes, "ok") if boxes else None


def parse_boxes(text: str, width: int, height: int, coord_mode: str = "norm1000") -> tuple[list[dict], str]:
    """Boxes in pixel xyxy plus a status: ok, empty, salvaged (partial JSON recovered) or fail.
    JSON answers get each box's character span from the order of `{...}` objects in the text."""
    raw_text = text
    if "{" not in text and "[" not in text:
        compact = parse_compact(_strip_reasoning(text), width, height, coord_mode)
        if compact is not None:
            return compact
    text = _strip_reasoning(text)
    items, status = None, "fail"
    for cand in _candidates(text):
        obj = _loads(cand)
        if obj is not None:
            items, status = _items(obj), "ok"
            break
    if items is None:
        recovered = [o for o in (_loads(m.group(0)) for m in OBJECT_RE.finditer(text)) if isinstance(o, dict)]
        if not recovered:
            return [], "fail"
        items, status = recovered, "salvaged"
    spans = [[m.start(), m.end()] for m in OBJECT_RE.finditer(raw_text)]
    spans = spans if len(spans) == len(items) else [None] * len(items)
    boxes = []
    for it, span in zip(items, spans):
        box = next((it[k] for k in BOX_KEYS if k in it), None)
        if not _is_box(box):
            continue
        label = next((it[k] for k in LABEL_KEYS if k in it), None)
        x1, y1, x2, y2 = to_pixels(box, width, height, coord_mode)
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            boxes.append({"label": label, "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                          "span": span})
    if status == "ok" and not boxes:
        status = "empty"
    return boxes, status


def normalize_label(label, label_to_category: dict[str, str]) -> str | None:
    if label is None:
        return None
    s = re.sub(r"[^a-z0-9 _-]", "", str(label).strip().lower()).replace("_", " ").strip()
    if s in label_to_category:
        return label_to_category[s]
    if s.endswith("s") and s[:-1] in label_to_category:
        return label_to_category[s[:-1]]
    return None


# --------------------------------------------------------------------------- backends

class TransformersBackend:
    def __init__(self, hf_id: str, class_name: str, dtype: str = "bfloat16", max_new_tokens: int = 1024,
                 chat_template_kwargs: dict | None = None, adapter: str | None = None, compile: bool = False):
        import torch
        import transformers

        self.torch = torch
        self.model_id = hf_id
        self.dtype = getattr(torch, dtype)
        self.processor = transformers.AutoProcessor.from_pretrained(hf_id)
        cls = getattr(transformers, class_name)
        # Whole model on the GPU. device_map="auto" lets accelerate offload modules to disk when memory is
        # tight, which buys nothing on unified-memory Jetsons and broke Cosmos3-Edge's vision encoder load.
        self.model = cls.from_pretrained(hf_id, dtype=self.dtype, device_map={"": 0}).eval()
        if adapter:                                         # LoRA from det_eval.vlm_finetune, merged for speed
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload().eval()
            self.model_id = f"{hf_id}+{adapter}"
        if compile:                                         # decode speed: static KV cache + CUDA-graph friendly forward
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=False)
        self.max_new_tokens = max_new_tokens
        self.template_kwargs = chat_template_kwargs or {}

    def generate(self, image_path: str | Path, prompt: str) -> tuple[str, int]:
        messages = [{"role": "user", "content": [{"type": "image", "path": str(image_path)},
                                                 {"type": "text", "text": prompt}]}]
        common = dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        try:
            inputs = self.processor.apply_chat_template(messages, **common, **self.template_kwargs)
        except TypeError:                                   # template does not accept the extra kwargs
            inputs = self.processor.apply_chat_template(messages, **common)
        inputs = inputs.to(self.model.device, self.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                      return_dict_in_generate=True, output_scores=True)
        new = out.sequences[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # log-probability of each generated token, and where its text lands in `text`
        logp = [self.torch.log_softmax(sc[0].float(), -1)[t].item() for sc, t in zip(out.scores, new[0])]
        tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        offsets, prev = [], ""
        for i in range(new.shape[1]):
            cur = tok.decode(new[0, : i + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            offsets.append((len(prev), len(cur)))
            prev = cur
        self.last_token_info = list(zip(offsets, logp))
        return text, int(new.shape[1])


class OpenAIBackend:
    """Minimal OpenAI-compatible chat client (stdlib only), e.g. for a vLLM server. No token scores."""

    last_token_info = None

    def __init__(self, base_url: str, model: str | None = None, max_tokens: int = 1024,
                 chat_template_kwargs: dict | None = None, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.max_tokens, self.timeout = max_tokens, timeout
        self.template_kwargs = chat_template_kwargs or {}
        self.model = model or self._get("/models")["data"][0]["id"]
        self.model_id = self.model

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base_url + path, timeout=self.timeout) as r:
            return json.loads(r.read())

    def generate(self, image_path: str | Path, prompt: str) -> tuple[str, int]:
        data = base64.b64encode(Path(image_path).read_bytes()).decode()
        mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0, "seed": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                    {"type": "text", "text": prompt}]}]}
        if self.template_kwargs:
            body["chat_template_kwargs"] = self.template_kwargs
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.loads(r.read())
        text = resp["choices"][0]["message"].get("content") or ""
        return text, int(resp.get("usage", {}).get("completion_tokens", 0))


# --------------------------------------------------------------------------- run

def draw(image_path, boxes: list[dict], gt_boxes: list[list[float]], out_path) -> None:
    """Predicted boxes (orange, labeled) over GT boxes (white): the coordinate-convention check."""
    from PIL import Image, ImageDraw

    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for x, y, w, h in gt_boxes:
        d.rectangle([x, y, x + w, y + h], outline=(255, 255, 255), width=1)
    for b in boxes:
        d.rectangle(b["bbox_xyxy"], outline=(235, 104, 52), width=2)
        d.text((b["bbox_xyxy"][0] + 2, max(0, b["bbox_xyxy"][1] - 11)), str(b["label"]), fill=(235, 104, 52))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=90)


def box_score(span, token_info) -> float:
    """Geometric-mean probability of the tokens that wrote this box (label and coordinates): a confidence a
    VLM never reports but its decoder computes anyway. 1.0 when no token information is available."""
    if not span or not token_info:
        return 1.0
    lp = [l for (a, b), l in token_info if b > span[0] and a < span[1]]
    return float(np.exp(np.mean(lp))) if lp else 1.0


def run_grounding(backend, gt: dict, image_root: str | Path, img_ids: list[int], prompts: list[str],
                  label_to_category: dict[str, str], prompt_template: str, coord_mode: str, out: str | Path,
                  draw_dir: str | Path | None = None, name: str | None = None) -> dict:
    images = {im["id"]: im for im in gt["images"]}
    cat_ids = {c["name"]: c["id"] for c in gt["categories"]}
    gt_by_img: dict[int, list] = {}
    for a in gt["annotations"]:
        if not a.get("iscrowd") and not a.get("ignore_iou"):
            gt_by_img.setdefault(a["image_id"], []).append(a["bbox"])
    prompt = prompt_template.format(categories=", ".join(prompts))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dets, status, unknown, latencies, tokens = [], Counter(), Counter(), [], []
    with open(out.with_suffix(".raw.jsonl"), "w") as raw:
        for n, img_id in enumerate(img_ids, 1):
            im = images[img_id]
            path = Path(image_root) / im["file_name"]
            t0 = time.perf_counter()
            backend.last_token_info = None
            text, n_tok = backend.generate(path, prompt)
            dt = time.perf_counter() - t0
            token_info = getattr(backend, "last_token_info", None)
            boxes, st = parse_boxes(text, im["width"], im["height"], coord_mode)
            status[st] += 1
            latencies.append(dt)
            tokens.append(n_tok)
            for b in boxes:
                cat = cat_ids.get(normalize_label(b["label"], label_to_category) or "")
                if cat is None:
                    unknown[str(b["label"])] += 1
                    continue
                x1, y1, x2, y2 = b["bbox_xyxy"]
                # score < 1 lets the evaluator compute AP/AR for a VLM; nudge exact 1.0 down so a scored run is
                # never mistaken for a score-less one
                sc = min(box_score(b.get("span"), token_info), 0.99999) if token_info else 1.0
                dets.append({"image_id": img_id, "category_id": cat,
                             "bbox": [x1, y1, round(x2 - x1, 2), round(y2 - y1, 2)], "score": round(sc, 5)})
            raw.write(json.dumps({"image_id": img_id, "status": st, "latency_s": round(dt, 3),
                                  "completion_tokens": n_tok, "text": text}) + "\n")
            if draw_dir:
                draw(path, boxes, gt_by_img.get(img_id, []), Path(draw_dir) / f"{img_id:06d}.jpg")
            print(f"  [{n}/{len(img_ids)}] {img_id:06d} {st:8s} {len(boxes):2d} boxes {dt:5.1f}s")

    save_json(dets, out)
    lat = np.array(latencies)
    tok = np.array(tokens, dtype=float)
    summary = {
        "name": name, "frames": len(img_ids), "parse_status": dict(status),
        "parse_ok_rate": (status["ok"] + status["empty"]) / max(len(img_ids), 1),
        "unknown_labels": dict(unknown.most_common(20)), "n_dets": len(dets),
        "latency_s": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)), "p90": float(np.percentile(lat, 90))},
        "completion_tokens_mean": float(tok.mean()),
        "decode_tok_per_s": float(tok.sum() / lat.sum()) if lat.sum() else None,
        "prompt": prompt, "coord_mode": coord_mode,
    }
    save_json(summary, out.with_suffix(".summary.json"), indent=2)
    write_manifest(out, name=name, backend=type(backend).__name__, model_id=getattr(backend, "model_id", None),
                   prompt=prompt, coord_mode=coord_mode, n_frames=len(img_ids))
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m det_eval.vlm_grounding", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--model", required=True, help="a `kind: vlm` entry in the config, e.g. cosmos3_nano")
    ap.add_argument("--backend", choices=["transformers", "openai"], default="transformers")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--dataset", default="kitti_val")
    ap.add_argument("--limit", type=int, help="first N frames of the subset (smoke test)")
    ap.add_argument("--draw", help="directory for images with parsed boxes drawn on them")
    ap.add_argument("--coord-mode", help="override vlm.coord_mode (norm1000, norm1, pixel, auto)")
    ap.add_argument("--out", help="default: runs/vlm/<model>__<dataset>__<n>.json")
    ap.add_argument("--split", help="frame-id file to run on instead of the dataset's VLM subset (e.g. the full val split)")
    ap.add_argument("--max-new-tokens", type=int, help="token budget (latency cap); default vlm.max_new_tokens")
    ap.add_argument("--compile", action="store_true", help="static KV cache + torch.compile for faster decoding")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    m, ds, vcfg, ov = cfg["models"][args.model], cfg["datasets"][args.dataset], cfg["vlm"], cfg["open_vocab"]
    gt = load_json(ds["gt_open_vocab"])
    img_ids = [int(i) for i in read_ids(args.split or ds["subset_vlm"])][: args.limit]
    if args.max_new_tokens:
        vcfg = {**vcfg, "max_new_tokens": args.max_new_tokens}
    if args.backend == "openai":
        backend = OpenAIBackend(args.base_url, None, vcfg["max_new_tokens"], vcfg.get("chat_template_kwargs"))
    else:
        backend = TransformersBackend(m["hf_id"], m["transformers_class"], max_new_tokens=vcfg["max_new_tokens"],
                                      chat_template_kwargs=vcfg.get("chat_template_kwargs"), adapter=m.get("adapter"),
                                      compile=args.compile)
    out = args.out or f"{cfg['paths']['runs']}/vlm/{args.model}__{args.dataset}__{len(img_ids)}.json"
    summary = run_grounding(backend, gt, ds["image_root"], img_ids, ov["prompts"], ov["label_to_category"],
                            m.get("prompt", vcfg["prompt"]), args.coord_mode or vcfg["coord_mode"], out, args.draw,
                            args.model)
    print(json.dumps({k: summary[k] for k in ("frames", "parse_status", "parse_ok_rate", "n_dets", "latency_s")}, indent=2))


if __name__ == "__main__":
    main()
