"""KITTI 2D object labels -> COCO-format ground truth, with KITTI-devkit ignore rules.

Expected layout (unzipped data_object_image_2.zip and data_object_label_2.zip):

    data/kitti/training/image_2/000000.png
    data/kitti/training/label_2/000000.txt

Usage:
    python -m det_eval.kitti gt --split configs/splits/kitti_val.txt --out data/kitti/annotations/kitti_val_coco.json
    python -m det_eval.kitti gt --split configs/splits/kitti_val.txt --open-vocab --out data/kitti/annotations/kitti_val_coco_openvocab.json
    python -m det_eval.kitti subset --split configs/splits/kitti_val.txt --n 300 --seed 0 --out configs/splits/kitti_val_vlm300.txt
    python -m det_eval.kitti stats --gt data/kitti/annotations/kitti_val_coco.json
    python -m det_eval.kitti copy --split configs/splits/kitti_val.txt \
        --image-zip ~/Downloads/data_object_image_2.zip --label-zip ~/Downloads/data_object_label_2.zip
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .common import image_size, load_json, load_yaml, now_iso, read_ids, save_json

# KITTI 2D benchmark difficulty definitions (cvlibs.net/datasets/kitti/eval_object.php):
# (name, minimum box height in px, maximum occlusion level, maximum truncation)
DIFFICULTIES = (
    ("easy", 40, 0, 0.15),
    ("moderate", 25, 1, 0.30),
    ("hard", 25, 2, 0.50),
)
LEVEL_NAMES = tuple(d[0] for d in DIFFICULTIES)


@dataclass(frozen=True)
class KittiObject:
    type: str
    truncated: float
    occluded: int
    alpha: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def parse_label_line(line: str) -> KittiObject:
    """One row of a KITTI label file: type, truncated, occluded, alpha, x1, y1, x2, y2, (3D fields...)."""
    f = line.split()
    if len(f) < 8:
        raise ValueError(f"malformed KITTI label line: {line!r}")
    return KittiObject(f[0], float(f[1]), int(float(f[2])), float(f[3]), *(float(v) for v in f[4:8]))


def parse_label_file(path: str | Path) -> list[KittiObject]:
    return [parse_label_line(line) for line in Path(path).read_text().splitlines() if line.strip()]


def difficulty_level(obj: KittiObject) -> int:
    """Easiest KITTI difficulty the object qualifies for: 0 easy, 1 moderate, 2 hard, -1 none.

    Levels are nested, as in the devkit: an easy object also counts for moderate and hard.
    """
    for i, (_, min_h, max_occ, max_trunc) in enumerate(DIFFICULTIES):
        if obj.height >= min_h and 0 <= obj.occluded <= max_occ and obj.truncated <= max_trunc:
            return i
    return -1


def build_coco_gt(
    kitti_root: str | Path,
    ids: list[str],
    mapping: dict,
    open_vocab: bool = False,
    image_dir: str = "image_2",
    label_dir: str = "label_2",
) -> dict:
    """COCO-format GT for the given frame ids.

    Extra per-annotation fields used by det_eval.evaluate:
      ignore_iou  1 for ignore regions matched with standard IoU (neighboring classes)
      iscrowd     1 for ignore regions matched by intersection over detection area (DontCare)
      level       easiest KITTI difficulty (0/1/2, -1 none); height in px; kitti_type
    """
    root = Path(kitti_root)
    name_to_id = {str(name): int(cid) for cid, name in mapping["coco_categories"].items()}
    categories = [
        {"id": cid, "name": name, "supercategory": "kitti"}
        for name, cid in sorted(name_to_id.items(), key=lambda kv: kv[1])
    ]
    extra = dict(mapping.get("open_vocab") or {}) if open_vocab else {}
    for name, spec in extra.items():
        name_to_id[name] = int(spec["id"])
        categories.append({"id": int(spec["id"]), "name": name, "supercategory": "open_vocab"})
    all_cat_ids = [c["id"] for c in categories]
    rules = mapping["kitti_classes"]

    images: list[dict] = []
    anns: list[dict] = []
    unknown: Counter = Counter()
    for sid in ids:
        img_rel = f"{image_dir}/{sid}.png"
        width, height = image_size(root / img_rel)
        image_id = int(sid)
        images.append({"id": image_id, "file_name": img_rel, "width": width, "height": height})

        for obj in parse_label_file(root / label_dir / f"{sid}.txt"):
            x1, y1 = max(0.0, obj.x1), max(0.0, obj.y1)
            x2, y2 = min(float(width), obj.x2), min(float(height), obj.y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            base = {
                "image_id": image_id,
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "kitti_type": obj.type,
                "truncated": obj.truncated,
                "occluded": obj.occluded,
                "height": round(bh, 2),
                "level": difficulty_level(obj),
            }

            def add(cat_id: int, iscrowd: int = 0, ignore_iou: int = 0) -> None:
                anns.append({**base, "id": len(anns) + 1, "category_id": cat_id,
                             "iscrowd": iscrowd, "ignore_iou": ignore_iou})

            for name, spec in extra.items():          # open-vocab classes become real GT
                if obj.type in spec.get("kitti", []):
                    add(name_to_id[name])

            rule = rules.get(obj.type)
            if rule is None:
                unknown[obj.type] += 1
                continue
            action = rule["action"]
            if action == "map":
                add(name_to_id[rule["coco"]])
            elif action == "ignore":
                targets = all_cat_ids if rule["for"] == "all" else [name_to_id[n] for n in rule["for"]]
                crowd = rule.get("match", "iou") == "crowd"
                for cid in targets:
                    add(cid, iscrowd=int(crowd), ignore_iou=int(not crowd))
            elif action != "drop":
                raise ValueError(f"unknown action {action!r} for KITTI class {obj.type}")

    info = {
        "description": "KITTI 2D object (training frames) in COCO format, devkit-style ignores",
        "created": now_iso(),
        "open_vocab": bool(open_vocab),
        "n_frames": len(images),
        "unknown_kitti_types": dict(unknown),
    }
    return {"info": info, "images": images, "annotations": anns, "categories": categories}


def copy_frames(ids: list[str], dst_root: str | Path, src_root: str | Path | None = None,
                image_zip: str | Path | None = None, label_zip: str | Path | None = None,
                image_dir: str = "image_2", label_dir: str = "label_2") -> int:
    """Copy only the listed frames (image + label) into dst_root/{image_2,label_2}.

    Reads either an unzipped KITTI `training/` folder (src_root) or the official zips directly
    (data_object_image_2.zip, data_object_label_2.zip), so the 12 GB image zip never has to be
    extracted on a small device.
    """
    import shutil
    import zipfile

    dst = Path(dst_root)
    for sub in (image_dir, label_dir):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    jobs = [(image_dir, ".png", image_zip), (label_dir, ".txt", label_zip)]
    for sub, ext, zpath in jobs:
        if zpath:
            with zipfile.ZipFile(zpath) as zf:
                for sid in ids:
                    out = dst / sub / f"{sid}{ext}"
                    if not out.exists():
                        with zf.open(f"training/{sub}/{sid}{ext}") as fsrc, open(out, "wb") as fdst:
                            shutil.copyfileobj(fsrc, fdst)
        elif src_root:
            for sid in ids:
                out = dst / sub / f"{sid}{ext}"
                if not out.exists():
                    shutil.copy2(Path(src_root) / sub / f"{sid}{ext}", out)
        else:
            raise ValueError(f"need src_root or a zip for {sub}")
    return len(ids)


def make_subset(ids: list[str], n: int, seed: int = 0) -> list[str]:
    """Fixed random subset (sorted) so every model is scored on the same frames."""
    ids = sorted(ids)
    if n >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, n))


def gt_stats(gt: dict) -> dict:
    """Counts of evaluable objects per class, KITTI difficulty and height bin (for the paper's data table)."""
    from .evaluate import HEIGHT_BINS, HEIGHT_LABELS

    names = {c["id"]: c["name"] for c in gt["categories"]}
    per_class: Counter = Counter()
    per_level: dict = defaultdict(Counter)
    per_bin: dict = defaultdict(Counter)
    ignored: Counter = Counter()
    for a in gt["annotations"]:
        name = names[a["category_id"]]
        if a.get("iscrowd") or a.get("ignore_iou"):
            ignored[name] += 1
            continue
        per_class[name] += 1
        lv = a.get("level", -1)
        per_level[name][LEVEL_NAMES[lv] if lv >= 0 else "none"] += 1
        h = a["bbox"][3]
        for (lo, hi), lbl in zip(HEIGHT_BINS, HEIGHT_LABELS):
            if lo <= h < hi:
                per_bin[name][lbl] += 1
                break
    return {
        "frames": len(gt["images"]),
        "objects": dict(per_class),
        "by_difficulty": {k: dict(v) for k, v in per_level.items()},
        "by_height_px": {k: dict(v) for k, v in per_bin.items()},
        "ignore_regions": dict(ignored),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m det_eval.kitti", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gt", help="build COCO-format GT for a split")
    g.add_argument("--kitti-root", default="data/kitti/training")
    g.add_argument("--split", required=True)
    g.add_argument("--mapping", default="configs/kitti_to_coco.yaml")
    g.add_argument("--open-vocab", action="store_true", help="add open-vocab categories (cyclist)")
    g.add_argument("--out", required=True)

    s = sub.add_parser("subset", help="fixed random subset of a split")
    s.add_argument("--split", required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)

    t = sub.add_parser("stats", help="object counts per class, difficulty and height bin")
    t.add_argument("--gt", required=True)

    c = sub.add_parser("copy", help="copy only a split's frames (from a folder or the official zips)")
    c.add_argument("--split", required=True)
    c.add_argument("--dst", default="data/kitti/training")
    c.add_argument("--src", help="unzipped KITTI training/ folder")
    c.add_argument("--image-zip", help="data_object_image_2.zip")
    c.add_argument("--label-zip", help="data_object_label_2.zip")

    args = ap.parse_args(argv)
    if args.cmd == "copy":
        n = copy_frames(read_ids(args.split), args.dst, args.src, args.image_zip, args.label_zip)
        print(f"copied {n} frames into {args.dst}")
        return
    if args.cmd == "gt":
        gt = build_coco_gt(args.kitti_root, read_ids(args.split), load_yaml(args.mapping), args.open_vocab)
        save_json(gt, args.out)
        print(f"wrote {args.out}: {len(gt['images'])} frames, {len(gt['annotations'])} annotations")
        if gt["info"]["unknown_kitti_types"]:
            print("unmapped KITTI types (skipped):", gt["info"]["unknown_kitti_types"])
    elif args.cmd == "subset":
        ids = make_subset(read_ids(args.split), args.n, args.seed)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(ids) + "\n")
        print(f"wrote {args.out}: {len(ids)} ids (seed {args.seed})")
    elif args.cmd == "stats":
        print(json.dumps(gt_stats(load_json(args.gt)), indent=2))


if __name__ == "__main__":
    main()
