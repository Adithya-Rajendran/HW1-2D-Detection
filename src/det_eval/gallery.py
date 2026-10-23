"""F6 failure gallery: hand-picked frames from the 300-frame subset, one per failure type.

Frames were picked by browsing the typed examples in results/failures.json and the audit sheets
(results/audit_unlabeled_small.json); the counts that go with each type are in the failure table.
GT is white (DontCare dashed blue), the model's kept boxes (conf >= 0.25; every VLM box) in its color,
and the box the panel is about is drawn thicker.

    python -m det_eval.gallery
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .common import REPO_ROOT, load_json, read_ids
from .evaluate import compose_cyclists

# (image_id, model, crop x1 y1 x2 y2, focus bbox xywh or None, title)
CASES = [
    (1376, "yolo26m_nms", (622, 64, 1242, 374), (956.7, 115.6, 284.3, 258.4),
     "(a) loose: near, truncated car covered by partial boxes"),
    (772, "yolo26m_nms", (380, 105, 760, 295), (449.3, 158.8, 212.5, 92.3),
     "(b) confused: COCO 'truck' on a KITTI Van"),
    (4202, "yolo26m_nms", (650, 118, 900, 243), (772.9, 178.9, 9.7, 22.6),
     "(c) hallucination: delineator post as a person"),
    (1683, "yolo26m_nms", (520, 115, 780, 245), (639.4, 176.3, 17.5, 14.1),
     "(d) real but unlabeled distant car, scored as a false positive"),
    (7469, "cosmos3_edge", (60, 95, 620, 375), (211.8, 150.4, 162.5, 209.7),
     "(e) Cosmos: riders boxed as pedestrians, cyclists missed"),
    (1801, "cosmos3_edge", (0, 95, 560, 375), (35.8, 197.7, 378.9, 176.4),
     "(f) Cosmos: large, near parked car never reported"),
]


def build(out: str | Path = "paper/figures/f6_gallery") -> Path:
    from . import plots

    gt = load_json("data/kitti/annotations/kitti_val_coco_openvocab.json")
    by_img = defaultdict(list)
    seen = set()
    for a in gt["annotations"]:
        key = (a["image_id"], tuple(a["bbox"]))
        if key not in seen:
            seen.add(key)
            by_img[a["image_id"]].append(a)
    sub = {int(i) for i in read_ids("configs/splits/kitti_val_vlm300.txt")}
    dets = {}
    d = [x for x in load_json("runs/preds/yolo26m_nms__kitti_val__640.json") if x["image_id"] in sub]
    dets["yolo26m_nms"] = [x for x in d + compose_cyclists(d) if x["score"] >= 0.25]
    dets["cosmos3_edge"] = load_json("runs/vlm/cosmos3_edge__kitti_val__300.json")
    per = {m: defaultdict(list) for m in dets}
    for m, lst in dets.items():
        for x in lst:
            per[m][x["image_id"]].append(x)
    files = {im["id"]: im["file_name"] for im in gt["images"]}
    cases = []
    for img, model, crop, focus, title in CASES:
        cases.append({
            "image": f"data/kitti/training/{files[img]}",
            "crop": list(crop),
            "gt": [a["bbox"] for a in by_img[img] if a["kitti_type"] != "DontCare"],
            "dontcare": [a["bbox"] for a in by_img[img] if a["kitti_type"] == "DontCare"],
            "preds": [{"model": model, "bbox": x["bbox"]} for x in per[model][img]],
            "focus": list(focus) if focus else None,
            "title": title,
        })
    return plots.fig_gallery(cases, out, ncols=2)


def main() -> None:
    import os

    os.chdir(REPO_ROOT)
    print(build())


if __name__ == "__main__":
    main()
