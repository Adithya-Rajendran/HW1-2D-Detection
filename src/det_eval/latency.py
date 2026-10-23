"""Batch-1 latency at a deployment threshold: end-to-end and per stage, versus objects per frame.

Images are decoded once before timing, so the numbers exclude disk reads and JPEG/PNG decoding.
GPU work is synchronized around every call. Peak memory comes from torch.cuda.max_memory_allocated():
nvidia-smi reports memory as "Not Supported" on GB10 (unified memory), and TensorRT engines allocate
outside PyTorch, so engine runs report peak memory as null.

Usage:
    python -m det_eval.latency --kind yolo --weights yolo26m.pt --nms false \
        --gt data/kitti/annotations/kitti_val_coco.json --image-root data/kitti/training \
        --out runs/latency/yolo26m_e2e__kitti_val__640.csv
    python -m det_eval.latency export --kind yolo --weights yolo26m.pt --nms false --name yolo26m_e2e
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np

from .common import load_json, save_json, write_manifest
from .predict_ultralytics import build_model, precision_kwargs, runtime_info


def valid_gt_counts(gt: dict) -> dict[int, int]:
    """Evaluable objects per image (ignore regions excluded): the x-axis of the latency-vs-clutter plot."""
    counts = {im["id"]: 0 for im in gt["images"]}
    for a in gt["annotations"]:
        if not a.get("iscrowd") and not a.get("ignore_iou"):
            counts[a["image_id"]] = counts.get(a["image_id"], 0) + 1
    return counts


def pick_images(gt: dict, n: int, seed: int = 0) -> list[dict]:
    images = sorted(gt["images"], key=lambda im: im["id"])
    if n >= len(images):
        return images
    idx = np.random.default_rng(seed).choice(len(images), n, replace=False)
    return [images[i] for i in sorted(idx)]


def measure(model, images: list[dict], image_root: str | Path, gt_counts: dict[int, int], imgsz: int, conf: float,
            iou: float, max_det: int, device, fp16: bool, extra: dict | None = None, warmup: int = 50):
    import cv2
    import torch

    arrays = [cv2.imread(str(Path(image_root) / im["file_name"])) for im in images]
    kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False,
              **precision_kwargs(fp16), **(extra or {}))
    cuda = torch.cuda.is_available() and str(device) != "cpu"
    for i in range(warmup):
        model.predict(arrays[i % len(arrays)], **kw)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for im, arr in zip(images, arrays):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(arr, **kw)[0]
        if cuda:
            torch.cuda.synchronize()
        e2e = (time.perf_counter() - t0) * 1e3
        rows.append({"image_id": im["id"], "n_gt": gt_counts.get(im["id"], 0), "n_det": len(r.boxes),
                     "e2e_ms": round(e2e, 3), **{f"{k}_ms": round(v, 3) for k, v in r.speed.items()}})
    peak_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None
    return rows, peak_mb


def summarize(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for key in ("e2e_ms", "preprocess_ms", "inference_ms", "postprocess_ms"):
        v = np.array([r[key] for r in rows if key in r], dtype=float)
        if v.size:
            out[key] = {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
                        "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99))}
    if "e2e_ms" in out:
        out["fps_mean"] = 1000.0 / out["e2e_ms"]["mean"]
    return out


def run_latency(kind: str, weights: str, gt_path: str, image_root: str, out: str, imgsz: int = 640,
                conf: float = 0.25, iou: float = 0.7, max_det: int = 300, device=0, fp16: bool = True,
                predict_kwargs: dict | None = None, build_yaml: str | None = None, prompts: list[str] | None = None,
                n_images: int = 500, warmup: int = 50, name: str | None = None) -> Path:
    gt = load_json(gt_path)
    model = build_model(kind, weights, build_yaml)
    if kind == "yoloe" and prompts and not str(weights).endswith(".engine"):
        model.set_classes(list(prompts))
    images = pick_images(gt, n_images)
    rows, peak_mb = measure(model, images, image_root, valid_gt_counts(gt), imgsz, conf, iou, max_det, device,
                            fp16, predict_kwargs, warmup)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"name": name, "weights": weights, "imgsz": imgsz, "conf": conf, "peak_mem_mb": peak_mb,
               "engine": str(weights).endswith(".engine"), **summarize(rows)}
    save_json(summary, out_path.with_suffix(".summary.json"), indent=2)
    write_manifest(out_path, name=name, kind=kind, weights=weights, gt=gt_path, imgsz=imgsz, conf=conf,
                   iou_nms=iou, max_det=max_det, fp16=fp16, predict_kwargs=predict_kwargs or {},
                   n_images=len(images), warmup=warmup, runtime=runtime_info(device))
    e2e = summary["e2e_ms"]
    print(f"{name or weights}: e2e p50 {e2e['p50']:.2f} ms, p99 {e2e['p99']:.2f} ms, peak {peak_mb} MB")
    return out_path


def export_engine(kind: str, weights: str, imgsz: int = 640, fp16: bool = True, nms: bool | None = None,
                  device=0, name: str | None = None, build_yaml: str | None = None,
                  prompts: list[str] | None = None, out_dir: str = "weights") -> Path:
    """TensorRT engine for a config. YOLO26 with nms=False exports the one-to-one (NMS-free) head."""
    model = build_model(kind, weights, build_yaml)
    if kind == "yoloe" and prompts:
        model.set_classes(list(prompts))
    kw = {"format": "engine", "imgsz": imgsz, "device": device, **precision_kwargs(fp16)}
    if nms is not None:
        kw["nms"] = nms
    path = Path(model.export(**kw))
    if name:
        dest = Path(out_dir) / f"{name}_{imgsz}.engine"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), dest)
        path = dest
    print(f"engine: {path}")
    return path


def _bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes")


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python -m det_eval.latency", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["yolo", "rtdetr", "yoloe"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--build-yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--nms", type=_bool)
    ap.add_argument("--prompts", nargs="+")
    if argv[:1] == ["export"]:
        ap.add_argument("--name")
        args = ap.parse_args(argv[1:])
        export_engine(args.kind, args.weights, args.imgsz, not args.fp32, args.nms, args.device, args.name,
                      args.build_yaml, args.prompts)
        return
    ap.add_argument("--gt", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args(argv)
    extra = {} if args.nms is None else {"nms": args.nms}
    run_latency(args.kind, args.weights, args.gt, args.image_root, args.out, args.imgsz, args.conf,
                device=args.device, fp16=not args.fp32, predict_kwargs=extra, build_yaml=args.build_yaml,
                prompts=args.prompts, n_images=args.n_images, warmup=args.warmup)


if __name__ == "__main__":
    main()
