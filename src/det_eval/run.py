"""Orchestrator: every stage of the experiment, driven by configs/experiments.yaml.

    python -m det_eval.run env                       # hardware/software report for the paper
    python -m det_eval.run gt                        # KITTI val GT (closed-set and open-vocab) in COCO format
    python -m det_eval.run predict [--only M ...] [--dataset D] [--imgsz N] [--force]
    python -m det_eval.run evaluate [--only M ...] [--bootstrap N] [--force]
    python -m det_eval.run latency [--only M ...] [--engine] [--n-images N]
    python -m det_eval.run vlm --model cosmos3_nano [--smoke] [--backend transformers|openai]
    python -m det_eval.run report
    python -m det_eval.run figures

Outputs: runs/preds/<model>__<dataset>__<imgsz>.json (+ .manifest.json, .timing.csv)
         runs/vlm/<model>__<dataset>__<frames>.json (+ .raw.jsonl, .summary.json)
         runs/latency/<model>__<dataset>__<imgsz>[__trt].csv (+ .summary.json)
         results/metrics/<model>__<dataset>__<imgsz|native>__<full|shared4|vlm300>.json
Every stage skips work whose output already exists unless --force is given.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .common import REPO_ROOT, load_json, load_yaml, read_ids, save_json

CLOSED_SET_SHARED = ["person", "car", "truck", "train"]   # COCO classes that exist in KITTI


def iter_runs(cfg: dict, only=None, dataset=None, imgsz=None):
    for exp in cfg["experiments"]:
        for model in exp["models"]:
            for size in exp["imgsz"]:
                if (only and model not in only) or (dataset and exp["dataset"] != dataset) or (imgsz and size != imgsz):
                    continue
                yield model, exp["dataset"], size


def pred_path(cfg, model, dataset, imgsz) -> Path:
    return Path(cfg["paths"]["runs"]) / "preds" / f"{model}__{dataset}__{imgsz}.json"


def metric_path(cfg, model, dataset, imgsz, evaluation) -> Path:
    return Path(cfg["paths"]["results"]) / "metrics" / f"{model}__{dataset}__{imgsz}__{evaluation}.json"


def gt_for(cfg, model: str, dataset: str) -> str:
    ds = cfg["datasets"][dataset]
    if cfg["models"][model].get("open_vocab") and "gt_open_vocab" in ds:
        return ds["gt_open_vocab"]
    return ds["gt"]


# --------------------------------------------------------------------------- stages

def cmd_env(cfg, args):
    from .envinfo import main as env_main
    env_main([])


def cmd_gt(cfg, args):
    from .kitti import build_coco_gt, gt_stats

    ds = cfg["datasets"]["kitti_val"]
    mapping = load_yaml("configs/kitti_to_coco.yaml")
    ids = read_ids("configs/splits/kitti_val.txt")
    for open_vocab, out in ((False, ds["gt"]), (True, ds["gt_open_vocab"])):
        if Path(out).exists() and not args.force:
            print(f"skip {out} (exists)")
            continue
        gt = build_coco_gt(ds["image_root"], ids, mapping, open_vocab)
        save_json(gt, out)
        print(f"wrote {out}: {len(gt['images'])} frames, {len(gt['annotations'])} annotations")
    stats = gt_stats(load_json(ds["gt_open_vocab"]))
    save_json(stats, Path(cfg["paths"]["results"]) / "kitti_val_stats.json", indent=2)
    print("objects:", stats["objects"])


def cmd_predict(cfg, args):
    from .predict_ultralytics import run_prediction

    d, ov = cfg["defaults"], cfg["open_vocab"]
    for model, dataset, size in iter_runs(cfg, args.only, args.dataset, args.imgsz):
        mc, ds = cfg["models"][model], cfg["datasets"][dataset]
        out = pred_path(cfg, model, dataset, size)
        if out.exists() and not args.force:
            print(f"skip {out} (exists)")
            continue
        print(f"== {model} on {dataset} @ {size}")
        run_prediction(mc["kind"], mc["weights"], gt_for(cfg, model, dataset), ds["image_root"], str(out), size,
                       conf=d["conf_ap"], iou=d["iou_nms"], max_det=d["max_det"], device=args.device or d["device"],
                       fp16=d["fp16"], predict_kwargs=mc.get("predict_kwargs"), build_yaml=mc.get("build_yaml"),
                       prompts=ov["prompts"] if mc["kind"] == "yoloe" else None,
                       label_to_category=ov["label_to_category"], conf_deploy=d["conf_deploy"], name=model)


def _evaluate(gt_path, dets, out: Path, args, d, **kw):
    from .evaluate import evaluate_predictions

    if out.exists() and not args.force:
        print(f"skip {out} (exists)")
        return
    res = evaluate_predictions(load_json(gt_path), dets, bootstrap=args.bootstrap, conf_deploy=d["conf_deploy"], **kw)
    res["gt"] = gt_path
    save_json(res, out, indent=2)
    coco = res.get("coco") or {}
    head = f"AP {coco['AP']:.3f}" if coco.get("AP") is not None else \
        f"F1@0.5 {res['operating_points']['iou0.5']['all_dets']['f1']:.3f}"
    print(f"wrote {out.name}: {head}")


def cmd_evaluate(cfg, args):
    from .evaluate import compose_cyclists

    d = cfg["defaults"]
    kitti = cfg["datasets"]["kitti_val"]
    subset = [int(i) for i in read_ids(kitti["subset_vlm"])]
    for model, dataset, size in iter_runs(cfg, args.only):
        pred = pred_path(cfg, model, dataset, size)
        if not pred.exists():
            print(f"missing {pred}, run predict first")
            continue
        dets = load_json(pred)
        open_vocab = cfg["models"][model].get("open_vocab", False)
        if dataset.startswith("coco"):
            _evaluate(cfg["datasets"][dataset]["gt"], dets, metric_path(cfg, model, dataset, size, "full"), args, d)
            _evaluate(cfg["datasets"][dataset]["gt"], dets, metric_path(cfg, model, dataset, size, "shared4"), args, d,
                      cat_names=CLOSED_SET_SHARED)
            continue
        if dataset.startswith("nuimages"):          # second driving domain: full, shared classes, cyclist
            ds = cfg["datasets"][dataset]
            _evaluate(gt_for(cfg, model, dataset), dets, metric_path(cfg, model, dataset, size, "full"), args, d)
            _evaluate(ds["gt"], dets, metric_path(cfg, model, dataset, size, "shared3"), args, d,
                      cat_names=["person", "car", "truck"])
            cyc = dets if open_vocab else dets + compose_cyclists(dets)
            _evaluate(ds["gt_open_vocab"], cyc, metric_path(cfg, model, dataset, size, "cyclist"), args, d,
                      cat_names=["cyclist"])
            continue
        _evaluate(gt_for(cfg, model, dataset), dets, metric_path(cfg, model, dataset, size, "full"), args, d)
        if cfg["models"][model].get("finetuned"):  # like-for-like with the zero-shot models' closed 4-class GT
            _evaluate(cfg["datasets"][dataset]["gt"], dets, metric_path(cfg, model, dataset, size, "closed4"), args, d)
        if not open_vocab:                        # rule-based cyclist on the full split, for the per-class table
            _evaluate(cfg["datasets"][dataset]["gt_open_vocab"], dets + compose_cyclists(dets),
                      metric_path(cfg, model, dataset, size, "cyclist"), args, d, cat_names=["cyclist"])
        if size == cfg["latency"]["imgsz"]:       # same frames, same GT and metrics as the VLM runs
            sub_dets = dets if open_vocab else dets + compose_cyclists(dets)
            _evaluate(kitti["gt_open_vocab"], sub_dets, metric_path(cfg, model, dataset, size, "vlm300"), args, d,
                      img_ids=subset)
    vlm_dir = Path(cfg["paths"]["runs"]) / "vlm"
    n_val = len(read_ids("configs/splits/kitti_val.txt"))
    for path in sorted(vlm_dir.glob("*__kitti_val__*.json")):
        parts = path.stem.split("__")
        if len(parts) != 3 or not parts[2].isdigit():
            continue
        model, n = parts[0], int(parts[2])
        if args.only and model not in args.only:
            continue
        if n == len(subset):                      # scored runs (token-probability confidences) also get AP/AR
            _evaluate(kitti["gt_open_vocab"], load_json(path), metric_path(cfg, model, "kitti_val", "native", "vlm300"),
                      args, d, img_ids=subset)
        elif n == n_val:
            _evaluate(kitti["gt_open_vocab"], load_json(path), metric_path(cfg, model, "kitti_val", "native", "full"),
                      args, d)


def cmd_latency(cfg, args):
    from .latency import export_engine, run_latency

    d, lc, ov = cfg["defaults"], cfg["latency"], cfg["open_vocab"]
    ds = cfg["datasets"]["kitti_val"]
    models = [m for m, mc in cfg["models"].items() if mc["kind"] != "vlm" and (not args.only or m in args.only)]
    for model in models:
        mc = cfg["models"][model]
        kwargs = dict(mc.get("predict_kwargs") or {})
        prompts = ov["prompts"] if mc["kind"] == "yoloe" else None
        weights = mc["weights"]
        suffix = ""
        if args.engine:
            engine = Path("weights") / f"{model}_{lc['imgsz']}.engine"
            if not engine.exists():
                export_engine(mc["kind"], weights, lc["imgsz"], d["fp16"], kwargs.get("nms"), args.device or d["device"],
                              model, mc.get("build_yaml"), prompts)
            weights, suffix = str(engine), "__trt"
        out = Path(cfg["paths"]["runs"]) / "latency" / f"{model}__kitti_val__{lc['imgsz']}{suffix}.csv"
        if out.exists() and not args.force:
            print(f"skip {out} (exists)")
            continue
        run_latency(mc["kind"], weights, ds["gt"], ds["image_root"], str(out), lc["imgsz"], d["conf_deploy"],
                    d["iou_nms"], d["max_det"], args.device or d["device"], d["fp16"], kwargs, mc.get("build_yaml"),
                    prompts, args.n_images or lc["n_images"], lc["warmup"], model)


def cmd_vlm(cfg, args):
    from .vlm_grounding import main as vlm_main

    argv = ["--model", args.model, "--backend", args.backend, "--base-url", args.base_url]
    if args.smoke:
        n = cfg["vlm"]["smoke_frames"]
        argv += ["--limit", str(n), "--draw", f"{cfg['paths']['runs']}/vlm_smoke/{args.model}"]
    vlm_main(argv)


def cmd_report(cfg, args):
    from .report import main as report_main
    report_main(["--results", f"{cfg['paths']['results']}/metrics", "--out-dir", cfg["paths"]["results"]])


def cmd_figures(cfg, args):
    from . import plots
    from .report import collect

    figs = Path(cfg["paths"]["figures"])
    df = collect(f"{cfg['paths']['results']}/metrics")
    if df.empty:
        print("no metrics yet: run evaluate first")
        return
    closed = [m for m in ("yolo26m_nms", "yolo26m_e2e", "rtdetr_l") if m in set(df["model"])]
    if not df[df["group"] == "height"].empty:
        print("F2", plots.fig_height_bins(df, figs / "f2_height_bins", closed + [m for m in ("yoloe26m", "yolo26m_ft_kitti") if m in set(df["model"])], cls="car"))
    if {"yolo26m_nms", "yolo26m_e2e"} <= set(df["model"]):
        print("F3", plots.fig_nms_gap(df, figs / "f3_nms_gap", classes=("car", "person", "truck", "train")))

    lat_dir = Path(cfg["paths"]["runs"]) / "latency"
    size = cfg["latency"]["imgsz"]
    def lat_file(m, ext):                 # prefer the TensorRT run: it is what a robot would deploy
        for suffix in ("__trt", ""):
            f = lat_dir / f"{m}__kitti_val__{size}{suffix}.{ext}"
            if f.exists():
                return f
        return None

    lat = {m: lat_file(m, "csv") for m in cfg["models"] if lat_file(m, "csv")}
    if lat:
        print("F4", plots.fig_latency_vs_clutter(lat, figs / "f4_latency_vs_clutter"))

    points = []
    for model, mc in cfg["models"].items():
        sub = df[(df["model"] == model) & (df["eval"] == "vlm300") & (df["group"] == "op")]
        if sub.empty:
            continue
        family = "vlm" if mc["kind"] == "vlm" else ("open" if mc.get("open_vocab") and not mc.get("finetuned") else "closed")
        kind = "all_dets" if family == "vlm" else "conf0.25"     # deployment operating point, as in the paper's table
        f1 = sub[(sub["metric"] == "f1@iou0.5") & (sub["key"] == kind)]["value"]
        if family == "vlm":
            summ = Path(cfg["paths"]["runs"]) / "vlm" / f"{model}__kitti_val__{len(read_ids(cfg['datasets']['kitti_val']['subset_vlm']))}.summary.json"
            latency_ms = load_json(summ)["latency_s"]["p50"] * 1000 if summ.exists() else None
        else:
            summ = lat_file(model, "summary.json")
            latency_ms = load_json(summ)["e2e_ms"]["p50"] if summ else None
        if len(f1) and latency_ms:
            points.append({"model": model, "latency_ms": latency_ms, "value": float(f1.iloc[0]), "family": family})
    if points:
        print("F5", plots.fig_speed_accuracy(points, figs / "f5_speed_accuracy"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m det_eval.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["env", "gt", "predict", "evaluate", "latency", "vlm", "report", "figures"])
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--only", nargs="+", help="restrict to these model names")
    ap.add_argument("--dataset")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--device")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--engine", action="store_true", help="latency: export and time TensorRT FP16 engines")
    ap.add_argument("--n-images", type=int)
    ap.add_argument("--model", help="vlm: model name from the config")
    ap.add_argument("--smoke", action="store_true", help="vlm: first N frames, boxes drawn for visual checking")
    ap.add_argument("--backend", default="transformers", choices=["transformers", "openai"])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    args = ap.parse_args(argv)

    os.chdir(REPO_ROOT)                     # all config paths are relative to the repo root
    cfg = load_yaml(args.config)
    if args.stage == "vlm" and not args.model:
        ap.error("vlm needs --model")
    globals()[f"cmd_{args.stage}"](cfg, args)


if __name__ == "__main__":
    main()
