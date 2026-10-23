"""Cross-check our COCO numbers against Ultralytics' own validator on the same images and weights.

Our pipeline runs `model.predict()` per image and scores with pycocotools. Ultralytics' published
numbers come from `model.val()`, which differs in post-processing (its NMS uses multi_label=True, so one
box can be emitted for several classes; `predict()` keeps one class per box) and in batching (rect
batches). Running both and scoring the validator's JSON with our evaluator separates "different
evaluator" from "different detections".

Usage:
    python -m det_eval.ultra_val labels        # YOLO-format labels + data yaml for data/coco (val2017)
    python -m det_eval.ultra_val val --weights yolo26m.pt --name yolo26m_nms
    python -m det_eval.ultra_val val --weights yolo26m.pt --name yolo26m_e2e --e2e
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .common import COCO80_TO_91, load_json, save_json, write_manifest

COCO_ROOT = Path("data/coco")


def write_labels(gt_path: Path = COCO_ROOT / "annotations/instances_val2017.json", root: Path = COCO_ROOT) -> Path:
    """Labels next to images/val2017 (Ultralytics swaps /images/ for /labels/), crowd boxes skipped."""
    gt = load_json(gt_path)
    to80 = {v: k for k, v in enumerate(COCO80_TO_91)}
    size = {im["id"]: (im["width"], im["height"], im["file_name"]) for im in gt["images"]}
    rows = defaultdict(list)
    for a in gt["annotations"]:
        if a.get("iscrowd"):
            continue
        w, h, _ = size[a["image_id"]]
        x, y, bw, bh = a["bbox"]
        if bw <= 0 or bh <= 0:
            continue
        rows[a["image_id"]].append(f"{to80[a['category_id']]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                                   f"{bw / w:.6f} {bh / h:.6f}")
    lbl_dir = root / "labels/val2017"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    listing = []
    for img_id, (_, _, fname) in size.items():
        (lbl_dir / (Path(fname).stem + ".txt")).write_text("\n".join(rows[img_id]) + ("\n" if rows[img_id] else ""))
        listing.append(f"./images/val2017/{fname}")
    # the validator only switches to COCO-91 ids and pycocotools when `val` ends with val2017.txt
    (root / "val2017.txt").write_text("\n".join(sorted(listing)) + "\n")
    names = load_json(gt_path)["categories"]
    by91 = {c["id"]: c["name"] for c in names}
    yaml = root / "coco_val.yaml"
    yaml.write_text(f"path: {root.resolve()}\ntrain: val2017.txt\nval: val2017.txt\nnames:\n" +
                    "\n".join(f"  {i}: {by91[COCO80_TO_91[i]]}" for i in range(80)) + "\n")
    return yaml


def validate(weights: str, name: str, e2e: bool, imgsz: int = 640, batch: int = 8, half: bool = True,
             data: str = str(COCO_ROOT / "coco_val.yaml"), out_dir: str = "runs/ultra_val") -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    kw = dict(data=data, imgsz=imgsz, batch=batch, conf=0.001, iou=0.7, half=half, save_json=True, plots=False,
              project=str(Path(out_dir).resolve()), name=name, exist_ok=True, device=0)
    m = model.val(**kw, **({"nms": False} if e2e else {}))   # nms=False selects the one-to-one head
    res = {"weights": weights, "e2e": e2e, "AP": float(m.box.map), "AP50": float(m.box.map50),
           "save_dir": str(m.save_dir)}
    save_json(res, Path(out_dir) / f"{name}.summary.json", indent=2)
    write_manifest(Path(out_dir) / f"{name}.summary.json", **kw)
    print(res)
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m det_eval.ultra_val", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("labels")
    v = sub.add_parser("val")
    v.add_argument("--weights", default="yolo26m.pt")
    v.add_argument("--name", required=True)
    v.add_argument("--e2e", action="store_true", help="one-to-one head (no NMS)")
    v.add_argument("--batch", type=int, default=8)
    v.add_argument("--fp32", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "labels":
        print("data yaml:", write_labels())
    else:
        validate(args.weights, args.name, args.e2e, batch=args.batch, half=not args.fp32)


if __name__ == "__main__":
    main()
