"""Run an Ultralytics detector (YOLO26, RT-DETR, YOLOE) over every image in a COCO-format GT file and
write COCO-format detections (original-image pixels) plus per-image timing and a manifest.

Usage (the orchestrator `python -m det_eval.run predict` calls the same function):
    python -m det_eval.predict_ultralytics --kind yolo --weights yolo26m.pt \
        --gt data/kitti/annotations/kitti_val_coco.json --image-root data/kitti/training \
        --imgsz 640 --out runs/preds/yolo26m_e2e__kitti_val__640.json --nms false
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

from .common import COCO80_TO_91, load_json, save_json, write_manifest


def build_model(kind: str, weights: str, build_yaml: str | None = None):
    if str(weights).endswith(".engine"):          # TensorRT engines load through the generic class
        from ultralytics import YOLO
        return YOLO(weights, task="detect")
    if kind == "yolo":
        from ultralytics import YOLO
        return YOLO(weights)
    if kind == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(weights)
    if kind == "yoloe":
        from ultralytics import YOLOE
        return YOLOE(build_yaml).load(weights) if build_yaml else YOLOE(weights)
    raise ValueError(f"unknown detector kind {kind!r}")


def precision_kwargs(fp16: bool) -> dict:
    """Ultralytics >= 8.4 replaced `half`/`int8` with `quantize`; support both."""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    if "quantize" in DEFAULT_CFG_DICT:
        return {"quantize": 16 if fp16 else 32}
    return {"half": bool(fp16)}


def category_map(model, kind: str, gt_categories: list[dict], prompts: list[str] | None = None,
                 label_to_category: dict[str, str] | None = None) -> dict[int, int]:
    """Model class index -> GT category id.

    Closed-set COCO models use the official 80 -> 91 id table, so every COCO class is written and the
    evaluator keeps only the classes present in the GT. YOLOE maps its text prompts through
    `label_to_category` (e.g. 'pedestrian' -> 'person', 'cyclist' -> 'cyclist').
    """
    if kind == "yoloe":
        if not prompts:
            raise ValueError("YOLOE needs text prompts")
        name_to_id = {c["name"]: c["id"] for c in gt_categories}
        mapping = {}
        for i, prompt in enumerate(prompts):
            name = (label_to_category or {}).get(prompt.lower(), prompt.lower())
            if name in name_to_id:
                mapping[i] = name_to_id[name]
        return mapping
    if len(model.names) != 80:              # fine-tuned model (e.g. KITTI 5-class): map by class name
        name_to_id = {c["name"]: c["id"] for c in gt_categories}
        mapping = {}
        for i, n in model.names.items():
            name = (label_to_category or {}).get(n.lower(), n.lower())
            if name in name_to_id:
                mapping[int(i)] = name_to_id[name]
        if not mapping:
            raise ValueError(f"no class of {model.names} maps to the GT categories")
        return mapping
    return {i: COCO80_TO_91[i] for i in range(80)}


def runtime_info(device) -> dict:
    info = {"python": platform.python_version(), "machine": platform.machine()}
    try:
        import torch
        import ultralytics

        info.update({"ultralytics": ultralytics.__version__, "torch": torch.__version__, "cuda": torch.version.cuda})
        if torch.cuda.is_available() and str(device) != "cpu":
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def force_multi_label(enabled: bool = True) -> None:
    """Make Ultralytics' predictor run NMS with multi_label=True, as its validator does.

    `predict()` keeps one class per box (argmax) before NMS; `val()` keeps every class above the threshold,
    so one box can be reported as both car and truck. The NMS-free YOLO26 head gets the same effect from
    its top-k over (anchor, class) pairs. This switch isolates that difference.
    """
    from ultralytics.utils import nms

    orig = getattr(nms.non_max_suppression, "_orig", nms.non_max_suppression)
    if not enabled:
        nms.non_max_suppression = orig
        return

    def patched(*args, **kwargs):
        kwargs["multi_label"] = True
        return orig(*args, **kwargs)

    patched._orig = orig
    nms.non_max_suppression = patched


def predict_dataset(model, gt: dict, image_root: str | Path, cat_map: dict[int, int], imgsz: int,
                    conf: float, iou: float, max_det: int, device, fp16: bool, extra: dict | None = None,
                    conf_deploy: float = 0.25, log_every: int = 500, batch: int = 1) -> tuple[list[dict], list[dict]]:
    """Detections (COCO results format) and per-image timing rows for every image in `gt`.

    One predict() call per image on purpose: Ultralytics turns a *list* of paths into in-memory PIL
    images and runs them as a single batch (autocast_list -> LoadPilAndNumpy), which would put all
    5,000 COCO images in memory at once on an 8 GB Jetson. Per-image calls also give true per-image timing.
    """
    images = gt["images"]
    extra = dict(extra or {})
    force_multi_label(bool(extra.pop("multi_label", False)))
    kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False, batch=batch,
              **precision_kwargs(fp16), **extra)
    dets, timing = [], []
    t0 = time.perf_counter()
    for n, im in enumerate(images, 1):
        r = model.predict(source=str(Path(image_root) / im["file_name"]), **kw)[0]
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy() if len(b) else []
        cls = b.cls.int().cpu().tolist() if len(b) else []
        scores = b.conf.cpu().tolist() if len(b) else []
        n_deploy = 0
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, scores):
            cat = cat_map.get(int(c))
            if cat is None:
                continue
            n_deploy += s >= conf_deploy
            dets.append({"image_id": im["id"], "category_id": cat,
                         "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                         "score": round(float(s), 5)})
        timing.append({"image_id": im["id"], **{f"{k}_ms": round(v, 3) for k, v in r.speed.items()},
                       "n_det": len(cls), "n_det_deploy": int(n_deploy)})
        if log_every and n % log_every == 0:
            print(f"  {n}/{len(images)} images, {time.perf_counter() - t0:.0f}s")
    return dets, timing


def run_prediction(kind: str, weights: str, gt_path: str, image_root: str, out: str, imgsz: int,
                   conf: float = 0.001, iou: float = 0.7, max_det: int = 300, device=0, fp16: bool = True,
                   predict_kwargs: dict | None = None, build_yaml: str | None = None,
                   prompts: list[str] | None = None, label_to_category: dict | None = None,
                   conf_deploy: float = 0.25, name: str | None = None) -> Path:
    gt = load_json(gt_path)
    model = build_model(kind, weights, build_yaml)
    if kind == "yoloe":
        model.set_classes(list(prompts))
    cat_map = category_map(model, kind, gt["categories"], prompts, label_to_category)
    t0 = time.perf_counter()
    dets, timing = predict_dataset(model, gt, image_root, cat_map, imgsz, conf, iou, max_det, device, fp16,
                                   predict_kwargs, conf_deploy)
    wall = time.perf_counter() - t0

    out_path = save_json(dets, out)
    with open(out_path.with_suffix(".timing.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(timing[0].keys()) if timing else ["image_id"])
        writer.writeheader()
        writer.writerows(timing)
    write_manifest(out_path, name=name, kind=kind, weights=weights, build_yaml=build_yaml, gt=gt_path,
                   image_root=image_root, imgsz=imgsz, conf=conf, iou_nms=iou, max_det=max_det, fp16=fp16,
                   predict_kwargs=predict_kwargs or {}, prompts=prompts, n_images=len(gt["images"]),
                   n_dets=len(dets), wall_s=round(wall, 1), runtime=runtime_info(device))
    print(f"wrote {out_path}: {len(dets)} detections on {len(gt['images'])} images in {wall:.0f}s")
    return out_path


def _bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m det_eval.predict_ultralytics", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["yolo", "rtdetr", "yoloe"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--build-yaml", help="YOLOE: detection-only model YAML to load the seg weights into")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--nms", type=_bool, help="YOLO26: false selects the NMS-free one-to-one head")
    ap.add_argument("--prompts", nargs="+", help="YOLOE text prompts")
    args = ap.parse_args(argv)
    extra = {} if args.nms is None else {"nms": args.nms}
    run_prediction(args.kind, args.weights, args.gt, args.image_root, args.out, args.imgsz, args.conf, args.iou,
                   args.max_det, args.device, not args.fp32, extra, args.build_yaml, args.prompts)


if __name__ == "__main__":
    main()
