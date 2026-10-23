"""LoRA fine-tuning of a grounding VLM (Cosmos 3 Edge) on KITTI train boxes, the tutorial's Qwen-LoRA recipe
(ngdet section 21) applied to Cosmos 3.

Targets use exactly the format the model already emits zero-shot (a fenced JSON list of
{"bbox_2d": [x1, y1, x2, y2], "label": ...} with coordinates on a 0-1000 grid), so evaluation is unchanged:
`python -m det_eval.vlm_grounding --model cosmos3_edge_ft` loads the adapter on top of the base model.

Label policy matches the YOLO fine-tune: Car/Pedestrian/Cyclist/Truck/Tram are targets under the prompt's
names (car, pedestrian, cyclist, truck, tram); Van, Person_sitting, Misc and DontCare get no box.
The loss covers only the answer tokens. Training frames are the same 3,412 used for YOLO; val is never seen.

Usage (DGX Spark / GX10; one GPU):
    python -m det_eval.vlm_finetune inspect                       # print LoRA target modules, no training
    python -m det_eval.vlm_finetune train --max-steps 50          # smoke test: loss falls, time per step
    python -m det_eval.vlm_finetune train --epochs 2              # full run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

from .common import image_size, load_yaml, read_ids, save_json, write_manifest
from .kitti import parse_label_file

KITTI_TO_PROMPT = {"Car": "car", "Pedestrian": "pedestrian", "Cyclist": "cyclist", "Truck": "truck", "Tram": "tram"}
SKIP_MODULES = re.compile(r"(vision|visual|image|patch|projector|mm_|merger|lm_head|embed)", re.I)


COMPACT_PROMPT = ("Locate every instance that belongs to the following categories: {categories}. "
                  "Answer one object per line as: label x1 y1 x2 y2, with coordinates from 0 to 1000, "
                  "nearest objects first. Answer none if there are none.")


def answer_text(objs, width: int, height: int, fmt: str = "json", order: str = "lr") -> str:
    """Training target. `json`: the fenced JSON the base model already produces (clean zero-shot comparison).
    `compact`: one `label x1 y1 x2 y2` per line, about a third of the tokens per box (speed).
    Order `lr` sorts left to right; `size` puts the largest (usually nearest) boxes first, so a capped token
    budget drops the far, small objects first."""
    items = []
    for o in objs:
        label = KITTI_TO_PROMPT.get(o.type)
        if label is None:
            continue
        x1, y1 = max(0.0, o.x1), max(0.0, o.y1)
        x2, y2 = min(float(width), o.x2), min(float(height), o.y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        box = [round(1000 * x1 / width), round(1000 * y1 / height), round(1000 * x2 / width), round(1000 * y2 / height)]
        items.append((box, label, (x2 - x1) * (y2 - y1)))
    if order == "size":
        items.sort(key=lambda t: (-t[2], t[0][0]))
    else:
        items.sort(key=lambda t: (t[0][0], t[0][1]))
    if fmt == "compact":
        return "\n".join(f"{lab} {b[0]} {b[1]} {b[2]} {b[3]}" for b, lab, _ in items) if items else "none"
    if not items:
        return "```json\n[]\n```"
    body = ",\n".join(f'    {{"bbox_2d": {json.dumps(b)}, "label": "{lab}"}}' for b, lab, _ in items)
    return f"```json\n[\n{body}\n]\n```"


def build_records(ids: list[str], kitti_root: str | Path, fmt: str = "json", order: str = "lr") -> list[dict]:
    root = Path(kitti_root)
    recs = []
    for sid in ids:
        img = root / "image_2" / f"{sid}.png"
        w, h = image_size(img)
        recs.append({"id": sid, "image": str(img), "answer": answer_text(parse_label_file(root / "label_2" / f"{sid}.txt"), w, h, fmt, order)})
    return recs


def lora_targets(model) -> tuple[str, list[str]]:
    """Regex over full module names selecting every nn.Linear of the language model (vision tower, projector,
    embeddings and head excluded), plus the distinct leaf names it covers, for the manifest."""
    import torch

    leaves = sorted({n.split(".")[-1] for n, m in model.named_modules()
                     if isinstance(m, torch.nn.Linear) and not SKIP_MODULES.search(n)})
    regex = r"^(?!.*(" + SKIP_MODULES.pattern[1:-1] + r")).*\.(" + "|".join(map(re.escape, leaves)) + r")$"
    return regex, leaves


def encode(processor, rec: dict, prompt: str, template_kwargs: dict):
    """input_ids / labels with the prompt (image + question) masked to -100, plus the image tensors."""
    user = {"role": "user", "content": [{"type": "image", "path": rec["image"]}, {"type": "text", "text": prompt}]}
    full = [user, {"role": "assistant", "content": [{"type": "text", "text": rec["answer"]}]}]
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    try:
        enc = processor.apply_chat_template(full, **kw, **template_kwargs)
        n_prompt = processor.apply_chat_template([user], add_generation_prompt=True, **kw, **template_kwargs)["input_ids"].shape[1]
    except TypeError:
        enc = processor.apply_chat_template(full, **kw)
        n_prompt = processor.apply_chat_template([user], add_generation_prompt=True, **kw)["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    enc["labels"] = labels
    return enc


def train(cfg_path: str, out_dir: str, epochs: float, max_steps: int | None, lr: float, rank: int, alpha: int,
          grad_accum: int, seed: int, train_ids: str, kitti_root: str, hf_id: str, log_every: int = 10,
          fmt: str = "json", order: str = "lr", val_ids: str | None = None, n_val: int = 100,
          val_every: int = 100) -> Path:
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model

    cfg = load_yaml(cfg_path)
    vcfg, ov = cfg["vlm"], cfg["open_vocab"]
    prompt = (COMPACT_PROMPT if fmt == "compact" else vcfg["prompt"]).format(categories=", ".join(ov["prompts"]))
    tkw = vcfg.get("chat_template_kwargs") or {}
    random.seed(seed)
    torch.manual_seed(seed)

    recs = build_records(read_ids(train_ids), kitti_root, fmt, order)
    val = build_records(read_ids(val_ids)[:n_val], kitti_root, fmt, order) if val_ids else []
    processor = transformers.AutoProcessor.from_pretrained(hf_id)
    model = transformers.AutoModelForImageTextToText.from_pretrained(hf_id, dtype=torch.bfloat16, device_map={"": 0})
    targets, leaves = lora_targets(model)
    print("LoRA on language-model linears:", leaves)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=targets,
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.train()

    steps_per_epoch = math.ceil(len(recs) / grad_accum)
    total = max_steps or int(math.ceil(epochs * steps_per_epoch))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")
    best = float("inf")

    def val_loss() -> float:
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for r in val:
                e = encode(processor, r, prompt, tkw).to(model.device)
                for k, v in e.items():
                    if torch.is_floating_point(v):
                        e[k] = v.to(torch.bfloat16)
                tot += model(**e).loss.item()
        model.train()
        return tot / max(len(val), 1)

    order, pos, step, t0, run_loss = [], 0, 0, time.perf_counter(), 0.0
    while step < total:
        for _ in range(grad_accum):
            if pos >= len(order):
                order, pos = random.sample(range(len(recs)), len(recs)), 0
            enc = encode(processor, recs[order[pos]], prompt, tkw).to(model.device)
            pos += 1
            for k, v in enc.items():
                if torch.is_floating_point(v):
                    enc[k] = v.to(torch.bfloat16)
            loss = model(**enc).loss / grad_accum
            loss.backward()
            run_loss += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % log_every == 0 or step == total:
            el = time.perf_counter() - t0
            row = {"step": step, "of": total, "loss": round(run_loss / log_every, 4), "lr": sched.get_last_lr()[0],
                   "s_per_step": round(el / step, 2), "eta_min": round(el / step * (total - step) / 60, 1)}
            print(row, flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            run_loss = 0.0
        if val and (step % val_every == 0 or step == total):
            vl = val_loss()
            row = {"step": step, "val_loss": round(vl, 4)}
            if vl < best:
                best = vl
                model.save_pretrained(out / "adapter_best")
                row["saved"] = "adapter_best"
            print(row, flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
    model.save_pretrained(out / "adapter")
    write_manifest(out / "adapter", hf_id=hf_id, epochs=epochs, steps=total, lr=lr, rank=rank, alpha=alpha,
                   grad_accum=grad_accum, seed=seed, train_ids=train_ids, n_frames=len(recs), targets=leaves,
                   format=fmt, order=order, val_ids=val_ids, n_val=len(val), best_val_loss=best,
                   prompt=prompt, wall_s=round(time.perf_counter() - t0, 1))
    return out / "adapter"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m det_eval.vlm_finetune", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inspect", "preview", "train"])
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--hf-id", default="nvidia/Cosmos3-Edge")
    ap.add_argument("--train-ids", default="configs/splits/kitti_train_ft.txt")
    ap.add_argument("--kitti-root", default="data/kitti/training")
    ap.add_argument("--out", default="runs/vlm_ft/cosmos3_edge_kitti")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=["json", "compact"], default="json")
    ap.add_argument("--order", choices=["lr", "size"], default="lr")
    ap.add_argument("--val-ids", default="configs/splits/kitti_train_minival.txt")
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=100)
    args = ap.parse_args(argv)
    if args.cmd == "preview":                          # targets for a few frames, no GPU
        for r in build_records(read_ids(args.train_ids)[:3], args.kitti_root, args.format, args.order):
            print(r["id"], r["answer"][:300], sep="\n")
        return
    if args.cmd == "inspect":
        import torch
        import transformers

        m = transformers.AutoModelForImageTextToText.from_pretrained(args.hf_id, dtype=torch.bfloat16)
        print("LoRA targets:", lora_targets(m))
        return
    print("adapter:", train(args.config, args.out, args.epochs, args.max_steps, args.lr, args.rank, args.alpha,
                            args.grad_accum, args.seed, args.train_ids, args.kitti_root, args.hf_id,
                            fmt=args.format, order=args.order, val_ids=args.val_ids, n_val=args.n_val,
                            val_every=args.val_every))


if __name__ == "__main__":
    main()
