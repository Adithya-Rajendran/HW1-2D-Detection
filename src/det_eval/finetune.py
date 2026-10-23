"""Fine-tune a COCO-pretrained Ultralytics detector on KITTI train and export the labels it needs.

The 3,712 train frames (Chen et al. split) are disjoint from the 3,769 val frames every model is scored on.
A fixed 300-frame `minival` is carved out of train for checkpoint selection and training curves, so the
val split is never seen during training or model selection.

Label policy (YOLO training has no ignore regions, so this differs from the evaluation GT on purpose):
    Car -> car, Pedestrian -> person, Cyclist -> cyclist, Truck -> truck, Tram -> train
    Van, Person_sitting, Misc, DontCare -> background (no box)
Evaluation still uses the devkit ignore rules (configs/kitti_to_coco.yaml), so a fine-tuned model is
not charged for firing on a Van or inside a DontCare region at test time.

Usage (inside the Jetson container, see scripts/docker_run.sh):
    python -m det_eval.finetune split                      # train_ft.txt + minival.txt from kitti_train.txt
    python -m det_eval.finetune export                     # data/kitti_yolo/{images,labels}/{train,minival}
    python -m det_eval.finetune train --hours 6            # Ultralytics train, time-boxed

Ablation (training with the devkit's neighbor classes as ignore regions, closer to how we evaluate):
    python -m det_eval.finetune export --out data/kitti_yolo_maskvan --mask Van Person_sitting
    python -m det_eval.finetune train --data data/kitti_yolo_maskvan/kitti5.yaml --name yolo26m_kitti5_640_maskvan --hours 7
Masked regions are filled with gray (114, Ultralytics' padding value) except where they overlap a kept,
labeled object, so the model never sees a Van as a negative example and never loses a labeled pixel.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .common import image_size, read_ids, save_json, write_manifest
from .kitti import make_subset, parse_label_file

# COCO names where a COCO class exists: Ultralytics copies the pretrained classifier rows for classes whose
# names match (4 of 5 here; only cyclist starts from scratch), which keeps more of the COCO head.
CLASSES = ["car", "person", "cyclist", "truck", "train"]
KITTI_TO_TRAIN = {"Car": 0, "Pedestrian": 1, "Cyclist": 2, "Truck": 3, "Tram": 4}


def split_train(train_ids: list[str], n_minival: int = 300, seed: int = 0) -> tuple[list[str], list[str]]:
    minival = make_subset(train_ids, n_minival, seed)
    held = set(minival)
    return [i for i in sorted(train_ids) if i not in held], minival


def yolo_lines(objs, width: int, height: int) -> list[str]:
    """KITTI objects -> YOLO 'cls cx cy w h' lines (normalized, clipped to the image)."""
    lines = []
    for o in objs:
        c = KITTI_TO_TRAIN.get(o.type)
        if c is None:
            continue
        x1, y1 = max(0.0, o.x1), max(0.0, o.y1)
        x2, y2 = min(float(width), o.x2), min(float(height), o.y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{c} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                     f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return lines


MASK_FILL = (114, 114, 114)


def masked_image(src: Path, objs, mask_types: set[str]):
    """Copy of the image with mask_types boxes grayed out, labeled objects of kept classes restored on top."""
    from PIL import Image

    im = Image.open(src).convert("RGB")
    orig = im.copy()
    w, h = im.size

    def box(o):
        return (max(0, int(o.x1)), max(0, int(o.y1)), min(w, int(round(o.x2)) + 1), min(h, int(round(o.y2)) + 1))

    n = 0
    for o in objs:
        if o.type in mask_types:
            b = box(o)
            if b[2] > b[0] and b[3] > b[1]:
                im.paste(MASK_FILL, b)
                n += 1
    for o in objs:
        if o.type in KITTI_TO_TRAIN:
            b = box(o)
            if b[2] > b[0] and b[3] > b[1]:
                im.paste(orig.crop(b), b[:2])
    return im, n


def export_split(ids: list[str], kitti_root: str | Path, out_root: str | Path, split: str,
                 mask_types: set[str] | None = None) -> dict:
    """Images (relative symlinks, or masked PNG copies when mask_types is set) plus one label file per frame."""
    kitti_root, out_root = Path(kitti_root), Path(out_root)
    img_dir, lbl_dir = out_root / "images" / split, out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    counts = [0] * len(CLASSES)
    n_masked = 0
    for sid in ids:
        src = kitti_root / "image_2" / f"{sid}.png"
        link = img_dir / f"{sid}.png"
        objs = parse_label_file(kitti_root / "label_2" / f"{sid}.txt")
        if mask_types:
            if not link.exists():
                im, n = masked_image(src, objs, mask_types)
                im.save(link)
                n_masked += n
        elif not link.exists():
            link.symlink_to(os.path.relpath(src.resolve(), img_dir.resolve()))
        w, h = image_size(src)
        lines = yolo_lines(objs, w, h)
        for ln in lines:
            counts[int(ln.split()[0])] += 1
        (lbl_dir / f"{sid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    out = {"frames": len(ids), "objects": dict(zip(CLASSES, counts))}
    if mask_types:
        out["masked_regions"] = n_masked
        out["mask_types"] = sorted(mask_types)
    return out


def write_data_yaml(out_root: str | Path) -> Path:
    out_root = Path(out_root).resolve()
    path = out_root / "kitti5.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    path.write_text(f"path: {out_root}\ntrain: images/train\nval: images/minival\nnames:\n{names}\n")
    return path


def train(model: str, data: str, hours: float, imgsz: int, batch: int, workers: int, name: str,
          project: str, seed: int = 0, **extra):
    from ultralytics import YOLO

    m = YOLO(model)
    res = m.train(data=data, imgsz=imgsz, batch=batch, workers=workers, time=hours, epochs=1000,
                  project=str(Path(project).resolve()), name=name, exist_ok=True, seed=seed,
                  amp=True, cache=False, plots=True, patience=1000, **extra)
    out = Path(project) / name
    write_manifest(out / "train.json", model=model, data=data, hours=hours, imgsz=imgsz, batch=batch,
                   workers=workers, seed=seed, extra=extra, save_dir=str(getattr(res, "save_dir", out)))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m det_eval.finetune", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--train", default="configs/splits/kitti_train.txt")
    s.add_argument("--n-minival", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("export")
    e.add_argument("--kitti-root", default="data/kitti/training")
    e.add_argument("--out", default="data/kitti_yolo")
    e.add_argument("--mask", nargs="+", help="KITTI types to gray out in training images (e.g. Van Person_sitting)")
    t = sub.add_parser("train")
    t.add_argument("--model", default="yolo26m.pt")
    t.add_argument("--data", default="data/kitti_yolo/kitti5.yaml")
    t.add_argument("--hours", type=float, default=6.0)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--name", default="yolo26m_kitti5_640")
    t.add_argument("--project", default="runs/finetune")
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.cmd == "split":
        ft, mv = split_train(read_ids(args.train), args.n_minival, args.seed)
        for name, ids in (("kitti_train_ft.txt", ft), ("kitti_train_minival.txt", mv)):
            Path("configs/splits", name).write_text("\n".join(ids) + "\n")
        print(f"train_ft {len(ft)} frames, minival {len(mv)} frames (seed {args.seed})")
    elif args.cmd == "export":
        stats = {}
        for split, ids_file in (("train", "configs/splits/kitti_train_ft.txt"),
                                ("minival", "configs/splits/kitti_train_minival.txt")):
            stats[split] = export_split(read_ids(ids_file), args.kitti_root, args.out, split,
                                        set(args.mask) if args.mask else None)
            print(split, stats[split])
        save_json(stats, Path(args.out) / "export_stats.json", indent=2)
        print("data yaml:", write_data_yaml(args.out))
    elif args.cmd == "train":
        print("run dir:", train(args.model, args.data, args.hours, args.imgsz, args.batch, args.workers,
                                args.name, args.project, args.seed))


if __name__ == "__main__":
    main()
