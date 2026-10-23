"""Count and type every error a model makes on KITTI, using the same matches pycocotools makes for AP.

Matching: IoU 0.5, detections at or above the deploy threshold (conf 0.25; VLM boxes have no score and
are all kept), devkit ignore rules from the GT file. Each error gets exactly one type, first match wins:

Missed objects (false negatives)
    confused     a kept detection of another class overlaps it (IoU >= 0.5): right place, wrong label
    loose        a same-class detection overlaps it, but only 0.1 <= IoU < 0.5: found, badly localized
    crowded      a neighboring same-class object (IoU >= 0.3) was detected and this one was not:
                 the signature of duplicate removal (NMS) or one box covering two objects
    small        under 25 px tall (about 50 m away for a pedestrian, KITTI focal length ~720 px)
    occluded     KITTI occlusion >= 2 or truncation > 0.3
    missed       none of the above

Wrong detections (false positives)
    duplicate    a same-class object it overlaps (IoU >= 0.5) was already matched by a higher-score box
    loose        overlaps a same-class object only at 0.1 <= IoU < 0.5
    confused     overlaps (IoU >= 0.5) a labeled object of another type (e.g. truck on a van, person on a cyclist)
    unlabeled_small   no labeled object nearby (IoU < 0.1) and under 25 px tall: often a real but unlabeled
                 distant object; see the manual audit in the paper
    background   no labeled object nearby
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from .common import load_json, read_ids, save_json
from .evaluate import coco_from_dict, gt_for_mode, load_dets, run_cocoeval

FN_TYPES = ("confused", "loose", "crowded", "small", "occluded", "missed")
FP_TYPES = ("duplicate", "loose", "confused", "unlabeled_small", "background")
SMALL_PX = 25.0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xywh boxes a (N,4) and b (M,4)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def _neighbor_found(overlaps: np.ndarray, hit: np.ndarray, self_index: int, thr: float = 0.3) -> bool:
    o = overlaps.copy()
    o[self_index] = 0.0
    return bool((o[hit] >= thr).any())


def analyze(gt: dict, dets: list[dict], img_ids=None, cat_names=None, conf: float = 0.25, iou: float = 0.5,
            n_examples: int = 40) -> dict:
    names = {c["id"]: c["name"] for c in gt["categories"]}
    cat_ids = sorted(k for k, v in names.items() if not cat_names or v in cat_names)
    kept = [d for d in dets if d["score"] >= conf and d["category_id"] in cat_ids]
    g = coco_from_dict(gt_for_mode(gt, "coco"))
    if img_ids is not None:
        img_ids = sorted(set(img_ids) & set(g.getImgIds()))
    E = run_cocoeval(g, load_dets(g, kept, "coco", cat_ids), cat_ids, img_ids, iou_thrs=[iou],
                     area_rng=[[0, 1e10]], area_lbl=["all"], max_dets=(1000,))
    by_img_obj = defaultdict(list)                     # every labeled KITTI object (one entry per object)
    seen = set()
    for a in gt["annotations"]:
        key = (a["image_id"], tuple(a["bbox"]), a["kitti_type"])
        if a["kitti_type"] != "DontCare" and key not in seen:
            seen.add(key)
            by_img_obj[a["image_id"]].append(a)
    dt_by_id = {d["id"]: d for d in E.cocoDt.dataset["annotations"]}
    dt_by_img = defaultdict(list)
    for d in E.cocoDt.dataset["annotations"]:
        dt_by_img[d["image_id"]].append(d)

    fn, fp = Counter(), Counter()
    fn_cls, fp_cls = defaultdict(Counter), defaultdict(Counter)
    tp = Counter()
    examples = {"fn": defaultdict(list), "fp": defaultdict(list)}
    fn_ids = set()
    for e in E.evalImgs:
        if e is None:
            continue
        img, cat = e["image_id"], e["category_id"]
        cname = names[cat]
        gts = [g.anns[i] for i in e["gtIds"]]
        gt_ign = np.asarray(e["gtIgnore"], bool)
        gt_matched = np.asarray(e["gtMatches"])[0] > 0
        dts = [dt_by_id[i] for i in e["dtIds"]]
        dt_matched = np.asarray(e["dtMatches"])[0] > 0
        dt_ign = np.asarray(e["dtIgnore"])[0].astype(bool)
        tp[cname] += int((dt_matched & ~dt_ign).sum())
        all_dt = dt_by_img[img]
        other_dt = np.array([d["bbox"] for d in all_dt if d["category_id"] != cat]).reshape(-1, 4)
        same_dt = np.array([d["bbox"] for d in dts]).reshape(-1, 4)
        valid = [k for k in range(len(gts)) if not gt_ign[k]]
        gt_boxes = np.array([gts[k]["bbox"] for k in valid]).reshape(-1, 4)
        gt_hit = gt_matched[valid]
        # misses
        for j, k in enumerate(valid):
            if gt_matched[k]:
                continue
            a = gts[k]
            box = np.array([a["bbox"]])
            near = (iou_matrix(box, same_dt)[0] >= 0.1) if len(same_dt) else np.zeros(0, bool)
            if len(other_dt) and iou_matrix(box, other_dt).max() >= iou:
                t = "confused"
            elif (near & ~dt_matched & ~dt_ign).any():
                t = "loose"                                  # a box was put here, but too far off
            elif near.any() or _neighbor_found(iou_matrix(box, gt_boxes)[0], gt_hit, j):
                t = "crowded"                                # the only nearby box went to a neighbor
            elif a["bbox"][3] < SMALL_PX:
                t = "small"
            elif a.get("occluded", 0) >= 2 or a.get("truncated", 0) > 0.3:
                t = "occluded"
            else:
                t = "missed"
            fn[t] += 1
            fn_cls[cname][t] += 1
            fn_ids.add(int(a["id"]))
            examples["fn"][t].append({"image_id": int(img), "class": cname, "bbox": a["bbox"], "gt_id": a["id"]})
        # wrong detections
        objs = by_img_obj[img]
        obj_boxes = np.array([o["bbox"] for o in objs]).reshape(-1, 4)
        for n, d in enumerate(dts):
            if dt_matched[n] or dt_ign[n]:
                continue
            box = np.array([d["bbox"]])
            o_same = iou_matrix(box, gt_boxes)[0] if len(gt_boxes) else np.zeros(0)
            if len(o_same) and (o_same >= iou).any():
                t = "duplicate"
            elif len(o_same) and (o_same >= 0.1).any():
                t = "loose"
            else:
                o_all = iou_matrix(box, obj_boxes)[0] if len(obj_boxes) else np.zeros(0)
                if len(o_all) and (o_all >= iou).any():
                    t = "confused"
                elif d["bbox"][3] < SMALL_PX and not (len(o_all) and (o_all >= 0.1).any()):
                    t = "unlabeled_small"
                else:
                    t = "background"
            fp[t] += 1
            fp_cls[cname][t] += 1
            examples["fp"][t].append({"image_id": int(img), "class": cname, "bbox": d["bbox"], "score": float(d.get("score", 1.0))})
    for side in examples:
        for t, lst in examples[side].items():
            key = (lambda x: -x["score"]) if side == "fp" else (lambda x: -x["bbox"][3])
            examples[side][t] = sorted(lst, key=key)[:n_examples]
    n_fn, n_fp = sum(fn.values()), sum(fp.values())
    return {"conf": conf, "iou": iou, "n_images": len(img_ids) if img_ids is not None else len(gt["images"]),
            "tp": sum(tp.values()), "fn": n_fn, "fp": n_fp,
            "fn_types": {t: fn[t] for t in FN_TYPES}, "fp_types": {t: fp[t] for t in FP_TYPES},
            "fn_by_class": {c: {t: v[t] for t in FN_TYPES} for c, v in fn_cls.items()},
            "fp_by_class": {c: {t: v[t] for t in FP_TYPES} for c, v in fp_cls.items()},
            "tp_by_class": dict(tp), "fn_gt_ids": sorted(fn_ids),
            "examples": {s: dict(v) for s, v in examples.items()}}


def recovered(a: dict, b: dict) -> dict:
    """Objects missed by A but found by B (and vice versa), from two analyze() results on the same GT."""
    fa, fb = set(a["fn_gt_ids"]), set(b["fn_gt_ids"])
    return {"missed_by_a_found_by_b": len(fa - fb), "missed_by_b_found_by_a": len(fb - fa), "missed_by_both": len(fa & fb)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m det_eval.failures", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-ids")
    ap.add_argument("--cats", nargs="+")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args(argv)
    ids = [int(i) for i in read_ids(args.img_ids)] if args.img_ids else None
    res = analyze(load_json(args.gt), load_json(args.dets), ids, args.cats, args.conf)
    save_json(res, args.out, indent=1)
    print("FN", res["fn"], res["fn_types"])
    print("FP", res["fp"], res["fp_types"])


if __name__ == "__main__":
    main()
