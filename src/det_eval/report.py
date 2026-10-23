"""Flatten results/metrics/*.json into tidy CSVs for the paper's tables and figures.

Metric files are named <model>__<dataset>__<imgsz|native>__<eval>.json, where eval is
full (all GT frames), shared4 (COCO restricted to the four KITTI-shared classes) or vlm300.

Usage:
    python -m det_eval.report            # writes results/summary_long.csv and results/summary_main.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from .common import load_json

MAIN_COLUMNS = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR100"]


def parse_name(stem: str) -> dict:
    parts = stem.split("__")
    if len(parts) != 4:
        raise ValueError(f"unexpected metrics file name {stem!r}")
    model, dataset, imgsz, evaluation = parts
    return {"model": model, "dataset": dataset, "imgsz": None if imgsz == "native" else int(imgsz), "eval": evaluation}


def _rows_for(res: dict, base: dict) -> list[dict]:
    rows = []

    def add(group: str, metric: str, value, key: str = "", ci=None):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        lo, hi = (ci if ci else (None, None))
        rows.append({**base, "group": group, "metric": metric, "key": key, "value": float(value), "lo": lo, "hi": hi})

    coco = res.get("coco") or {}
    for m in MAIN_COLUMNS:
        add("coco", m, coco.get(m), ci=coco.get(f"{m}_ci95"))
    for name, v in (coco.get("per_class_AP") or {}).items():
        add("coco_class", "AP", v, name)
    for name, v in (coco.get("per_class_AP50") or {}).items():
        add("coco_class", "AP50", v, name)
    for lbl, entry in (res.get("height_bins") or {}).items():
        add("height", "AP", entry.get("AP"), lbl, entry.get("AP_ci95"))
        add("height", "AP50", entry.get("AP50"), lbl)
        add("height", "recall_all_dets", entry.get("recall_iou0.5_all_dets"), lbl)
        add("height", "n_gt", entry.get("n_gt"), lbl)
        for name, v in (entry.get("per_class_AP") or {}).items():
            add("height_class", "AP", v, f"{lbl}|{name}")
    for level, per_cls in ((res.get("kitti") or {}).get("AP_R40") or {}).items():
        for name, v in per_cls.items():
            add("kitti", "AP_R40", v, f"{level}|{name}")
    for iou_key, ops in (res.get("operating_points") or {}).items():
        for kind, op in ops.items():
            if kind == "per_class_all_dets":
                for name, o in op.items():
                    for m in ("precision", "recall", "f1"):
                        add("op_class", f"{m}@{iou_key}", o.get(m), name)
                continue
            if not op:
                continue
            for m in ("precision", "recall", "f1", "score_thr"):
                add("op", f"{m}@{iou_key}", op.get(m), kind)
    return rows


def collect(results_dir: str | Path = "results/metrics") -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_dir).glob("*__*__*__*.json")):
        if path.name.endswith(".manifest.json"):
            continue
        rows += _rows_for(load_json(path), parse_name(path.stem))
    return pd.DataFrame(rows)


def main_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model x dataset x imgsz (full evaluation): the paper's main results table."""
    sub = df[(df["group"] == "coco") & (df["eval"] == "full")]
    if sub.empty:
        return sub
    return (sub.pivot_table(index=["dataset", "imgsz", "model"], columns="metric", values="value", dropna=False)
            .reindex(columns=MAIN_COLUMNS).round(4).reset_index())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m det_eval.report", description=__doc__)
    ap.add_argument("--results", default="results/metrics")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)
    df = collect(args.results)
    if df.empty:
        print(f"no metric files in {args.results}")
        return
    out = Path(args.out_dir)
    df.to_csv(out / "summary_long.csv", index=False)
    table = main_table(df)
    table.to_csv(out / "summary_main.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
