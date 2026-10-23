"""End-to-end on synthetic data: evaluate -> metric files -> report -> every figure renders."""

import numpy as np
import pandas as pd

from det_eval import plots
from det_eval.common import save_json
from det_eval.evaluate import evaluate_predictions
from det_eval.kitti import build_coco_gt
from det_eval.report import collect, main_table

BOXES = [(1, 3, 100, 150, 300, 300), (1, 1, 700, 120, 740, 220), (2, 3, 100, 150, 200, 200),
         (2, 3, 500, 180, 530, 200), (2, 8, 900, 100, 1200, 300)]


def _dets(drop_small: bool, jitter: float, seed: int):
    rng = np.random.default_rng(seed)
    out = []
    for img, cat, x1, y1, x2, y2 in BOXES:
        if drop_small and y2 - y1 < 25:
            continue
        j = rng.normal(0, jitter, 4)
        out.append({"image_id": img, "category_id": cat, "bbox": [x1 + j[0], y1 + j[1], x2 - x1 + j[2], y2 - y1 + j[3]],
                    "score": float(rng.uniform(0.3, 0.95))})
    return out


def test_report_and_figures(kitti_root, mapping, tmp_path):
    gt = build_coco_gt(kitti_root, ["000001", "000002"], mapping)
    metrics = tmp_path / "metrics"
    for i, model in enumerate(["yolo26m_nms", "yolo26m_e2e", "rtdetr_l"]):
        for size in (640, 1280):
            res = evaluate_predictions(gt, _dets(size == 640, 2.0 + i, i), bootstrap=50)
            save_json(res, metrics / f"{model}__kitti_val__{size}__full.json")
        res = evaluate_predictions(gt, _dets(False, 1.0, i), bootstrap=0, cat_names=["person", "car", "truck", "train"])
        save_json(res, metrics / f"{model}__coco_val2017__640__shared4.json")

    df = collect(metrics)
    assert {"coco", "height", "kitti", "op"} <= set(df["group"])
    table = main_table(df)
    assert len(table) == 6 and "AP" in table.columns

    figs = tmp_path / "figs"
    assert plots.fig_height_bins(df, figs / "f2", ["yolo26m_nms", "yolo26m_e2e", "rtdetr_l"]).exists()
    assert plots.fig_nms_gap(df, figs / "f3").exists()

    lat = {}
    for i, model in enumerate(["yolo26m_nms", "yolo26m_e2e"]):
        n = np.arange(40) % 14
        pd.DataFrame({"n_gt": n, "e2e_ms": 20 + i * 3 + n * (0.4 - 0.35 * i)}).to_csv(tmp_path / f"{model}.csv", index=False)
        lat[model] = tmp_path / f"{model}.csv"
    assert plots.fig_latency_vs_clutter(lat, figs / "f4").exists()

    points = [{"model": "yolo26m_nms", "latency_ms": 22.0, "value": 0.71, "family": "closed"},
              {"model": "yoloe26m", "latency_ms": 30.0, "value": 0.66, "family": "open"},
              {"model": "cosmos3_edge", "latency_ms": 31000.0, "value": 0.42, "family": "vlm"}]
    assert plots.fig_speed_accuracy(points, figs / "f5").exists()

    cases = [{"image": kitti_root / "image_2" / "000001.png", "crop": [600, 100, 900, 300],
              "gt": [[700, 120, 40, 100]], "preds": [{"model": "yolo26m_nms", "bbox": [702, 118, 40, 104]}],
              "title": "000001 pedestrian"}]
    assert plots.fig_gallery(cases, figs / "f6", ncols=1, width=3.5).exists()
