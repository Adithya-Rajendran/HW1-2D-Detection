"""nuImages 2D boxes -> COCO-format ground truth with the same ignore semantics as our KITTI GT.

nuImages (Motional, CC BY-NC-SA 4.0) labels 2D boxes on key frames from six cameras at 1600x900. It is the
second domain for the cross-dataset claims: the course tutorial recommends it as the reliable 2D driving
benchmark. Mapping lives in configs/nuimages_to_coco.yaml (same action/ignore vocabulary as KITTI).

Expected layout (official devkit download, extracted):
    data/nuimages/v1.0-val/{sample_data,object_ann,category,attribute}.json
    data/nuimages/samples/CAM_*/...jpg

Usage:
    python -m det_eval.nuimages subset --n 3000 --seed 0 --out configs/splits/nuimages_val3k.txt
    python -m det_eval.nuimages gt --split configs/splits/nuimages_val3k.txt --out data/nuimages_coco/nuimages_val3k_coco.json
    python -m det_eval.nuimages gt --split configs/splits/nuimages_val3k.txt --open-vocab --out data/nuimages_coco/nuimages_val3k_coco_openvocab.json
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from .common import load_json, load_yaml, now_iso, read_ids, save_json

ROOT = Path("data/nuimages")


def load_tables(root: Path = ROOT, version: str = "v1.0-val") -> dict:
    t = {name: load_json(root / version / f"{name}.json") for name in ("sample_data", "object_ann", "category", "attribute")}
    return t


def key_frames(tables: dict) -> list[dict]:
    return [sd for sd in tables["sample_data"] if sd.get("is_key_frame")]


def make_subset(tables: dict, n: int, seed: int = 0) -> list[str]:
    """Sorted sample_data tokens of n random key frames."""
    toks = sorted(sd["token"] for sd in key_frames(tables))
    return sorted(random.Random(seed).sample(toks, n)) if n < len(toks) else toks


def _rule(name: str, attrs: set[str], rules: dict):
    """First rule whose category matches (exact or prefix ending in '.*') and whose attribute condition holds."""
    for r in rules:
        cat = r["category"]
        hit = name == cat or (cat.endswith(".*") and name.startswith(cat[:-1]))
        if hit and all(a in attrs for a in r.get("with_attributes", [])) and \
                not any(a in attrs for a in r.get("without_attributes", [])):
            return r
    return None


def build_coco_gt(tables: dict, tokens: list[str], mapping: dict, open_vocab: bool = False) -> dict:
    cats = [{"id": int(cid), "name": name, "supercategory": "nuimages"} for cid, name in mapping["coco_categories"].items()]
    name_to_id = {c["name"]: c["id"] for c in cats}
    if open_vocab:
        for name, spec in (mapping.get("open_vocab") or {}).items():
            cats.append({"id": int(spec["id"]), "name": name, "supercategory": "open_vocab"})
            name_to_id[name] = int(spec["id"])
    all_ids = [c["id"] for c in cats]
    cat_name = {c["token"]: c["name"] for c in tables["category"]}
    attr_name = {a["token"]: a["name"] for a in tables["attribute"]}
    sd_by_tok = {sd["token"]: sd for sd in tables["sample_data"]}
    anns_by_sd: dict[str, list] = {}
    for oa in tables["object_ann"]:
        anns_by_sd.setdefault(oa["sample_data_token"], []).append(oa)

    images, anns, unknown = [], [], Counter()
    for i, tok in enumerate(tokens, start=1):
        sd = sd_by_tok[tok]
        images.append({"id": i, "file_name": sd["filename"], "width": sd["width"], "height": sd["height"], "token": tok})
        for oa in anns_by_sd.get(tok, []):
            name = cat_name[oa["category_token"]]
            attrs = {attr_name[a] for a in oa.get("attribute_tokens", [])}
            x1, y1, x2, y2 = (float(v) for v in oa["bbox"])
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            base = {"image_id": i, "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2), "kitti_type": name, "src_type": name, "height": round(bh, 2),
                    "level": -1, "truncated": 0.0, "occluded": 0}

            def add(cid, iscrowd=0, ignore_iou=0):
                anns.append({**base, "id": len(anns) + 1, "category_id": cid, "iscrowd": iscrowd, "ignore_iou": ignore_iou})

            if open_vocab:
                for ov_name, spec in (mapping.get("open_vocab") or {}).items():
                    if _rule(name, attrs, spec["rules"]):
                        add(name_to_id[ov_name])
            r = _rule(name, attrs, mapping["rules"])
            if r is None:
                unknown[name] += 1
                continue
            if r["action"] == "map":
                add(name_to_id[r["coco"]])
            elif r["action"] == "ignore":
                targets = all_ids if r["for"] == "all" else [name_to_id[n] for n in r["for"] if n in name_to_id]
                for cid in targets:
                    add(cid, iscrowd=int(r.get("match", "iou") == "crowd"), ignore_iou=int(r.get("match", "iou") == "iou"))
    info = {"description": "nuImages val key frames in COCO format, ignore rules from configs/nuimages_to_coco.yaml",
            "created": now_iso(), "open_vocab": bool(open_vocab), "n_frames": len(images), "unmapped": dict(unknown),
            "dataset": "nuimages"}
    return {"info": info, "images": images, "annotations": anns, "categories": cats}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m det_eval.nuimages", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--version", default="v1.0-val")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--n", type=int, default=3000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    g = sub.add_parser("gt")
    g.add_argument("--split", required=True)
    g.add_argument("--mapping", default="configs/nuimages_to_coco.yaml")
    g.add_argument("--open-vocab", action="store_true")
    g.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    tables = load_tables(Path(args.root), args.version)
    if args.cmd == "subset":
        toks = make_subset(tables, args.n, args.seed)
        Path(args.out).write_text("\n".join(toks) + "\n")
        print(f"wrote {args.out}: {len(toks)} key frames of {len(key_frames(tables))} (seed {args.seed})")
    else:
        gt = build_coco_gt(tables, read_ids(args.split), load_yaml(args.mapping), args.open_vocab)
        save_json(gt, args.out)
        print(f"wrote {args.out}: {len(gt['images'])} frames, {len(gt['annotations'])} annotations; unmapped {gt['info']['unmapped']}")


if __name__ == "__main__":
    main()
