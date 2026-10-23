"""Paper figures. One color per model in every figure (fixed slot order, never re-ranked),
solid hairline grids, a legend whenever there are two or more series, and vector PDF output.

Palette: the validated categorical order from the dataviz reference palette (light mode).
Slots 3-5 sit below 3:1 contrast on white, so every figure has a table twin in the paper
(results/summary_*.csv) and the scatter carries direct labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODEL_COLORS = {                 # fixed slot per entity
    "yolo26m_nms": "#2a78d6",    # 1 blue
    "yolo26m_e2e": "#eb6834",    # 2 orange
    "rtdetr_l": "#1baf7a",       # 3 aqua
    "yoloe26m": "#eda100",       # 4 yellow
    "cosmos3_nano": "#e87ba4",   # 5 magenta
    "qwen3vl_8b": "#008300",     # 6 green
    "cosmos3_edge": "#6250d6",   # 7 violet
    "yolo26m_nms_ml": "#e34948", # 8 red
    "yolo26m_ft_kitti": "#008300",      # 6 green (Qwen3-VL-8B was not run)
    "yolo26m_ft_kitti_e2e": "#e87ba4",  # 5 magenta (Cosmos 3 Nano was not run)
}
MODEL_LABELS = {
    "yolo26m_nms": "YOLO26m (NMS)",
    "yolo26m_e2e": "YOLO26m (NMS-free)",
    "rtdetr_l": "RT-DETR-l",
    "yoloe26m": "YOLOE-26m",
    "cosmos3_nano": "Cosmos 3 Nano",
    "qwen3vl_8b": "Qwen3-VL-8B",
    "cosmos3_edge": "Cosmos 3 Edge",
    "yolo26m_nms_ml": "YOLO26m (NMS, multi-label)",
    "yolo26m_ft_kitti": "YOLO26m fine-tuned (NMS)",
    "yolo26m_ft_kitti_e2e": "YOLO26m fine-tuned (NMS-free)",
}
FAMILY_MARKER = {"closed": "o", "open": "s", "vlm": "D"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLUMN_W, PAGE_W = 3.5, 7.16     # IEEE column and page widths (inches)

HEIGHT_ORDER = ["<25", "25-40", "40-80", ">=80"]
CLUTTER_BINS = [(0, 2, "0-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 10**6, "11+")]


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "lines.linewidth": 1.4, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 4.5, "lines.markeredgewidth": 0.8, "lines.markeredgecolor": "white",
        "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def _save(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out.with_suffix(".pdf")


def _err(sub: pd.DataFrame):
    if sub["lo"].isna().all():
        return None
    return np.vstack([sub["value"] - sub["lo"], sub["hi"] - sub["value"]]).clip(min=0)


def fig_height_bins(df: pd.DataFrame, out: str | Path, models: list[str], metric: str = "AP",
                    dataset: str = "kitti_val", sizes=(640, 1280), cls: str | None = "car") -> Path:
    """F2: AP by box height, one panel per input size. Answers RQ1 (resolution vs appearance).

    Defaults to one class (car). The class-averaged bins mix in classes with almost no small objects
    (KITTI val has 10 trams and 32 pedestrians under 25 px), so their AP there is noise.
    """
    style()
    if cls:
        d = df[(df["group"] == "height_class") & (df["metric"] == metric) & (df["dataset"] == dataset)
               & (df["eval"] == "full") & df["key"].str.endswith(f"|{cls}")].copy()
        d["key"] = d["key"].str.split("|").str[0]
    else:
        d = df[(df["group"] == "height") & (df["metric"] == metric) & (df["dataset"] == dataset) & (df["eval"] == "full")]
    fig, axes = plt.subplots(1, len(sizes), figsize=(COLUMN_W, 1.9), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(HEIGHT_ORDER))
    dodge = np.linspace(-0.09, 0.09, max(len(models), 2))      # keeps error bars from overlapping
    for ax, size in zip(axes, sizes):
        for j, m in enumerate(models):
            sub = d[(d["model"] == m) & (d["imgsz"] == size)].set_index("key").reindex(HEIGHT_ORDER).reset_index()
            if sub["value"].isna().all():
                continue
            ax.errorbar(x + dodge[j], sub["value"], yerr=_err(sub), color=MODEL_COLORS[m], marker="o", capsize=0,
                        elinewidth=0.8, label=MODEL_LABELS[m])
        ax.set_title(f"input {size} px", color=INK_2)
        ax.set_xticks(x, HEIGHT_ORDER)
        ax.set_xlabel("GT box height (px)")
    axes[0].set_ylim(bottom=0)          # after all panels: shared y-limits must cover every series
    axes[0].set_ylabel(f"KITTI {metric}" + (f" ({cls})" if cls else ""))
    _legend_below(fig, axes[0])
    return _save(fig, out)


def _legend_below(fig, ax, ncol: int = 2) -> None:
    """Legend under the x-axis labels (tight_layout first so the labels sit inside the canvas)."""
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.0), handlelength=1.6,
               columnspacing=1.2)


def fig_nms_gap(df: pd.DataFrame, out: str | Path, classes=("car", "person", "truck"), imgsz: int = 640) -> Path:
    """F3: AP(NMS head) - AP(NMS-free head), same weights, in-domain vs KITTI. Answers RQ2."""
    style()

    def gap(dataset: str, evaluation: str) -> dict:
        d = df[(df["group"] == "coco_class") & (df["metric"] == "AP") & (df["dataset"] == dataset)
               & (df["eval"] == evaluation) & (df["imgsz"] == imgsz)]
        a = d[d["model"] == "yolo26m_nms"].set_index("key")["value"]
        b = d[d["model"] == "yolo26m_e2e"].set_index("key")["value"]
        return {c: 100 * (a.get(c, np.nan) - b.get(c, np.nan)) for c in classes}

    coco, kitti = gap("coco_val2017", "shared4"), gap("kitti_val", "full")
    fig, ax = plt.subplots(figsize=(COLUMN_W, 0.35 + 0.32 * len(classes)))
    y = np.arange(len(classes))[::-1]
    for yi, c in zip(y, classes):
        ax.plot([coco[c], kitti[c]], [yi, yi], color=AXIS, linewidth=1.0, zorder=1)
    ax.scatter([coco[c] for c in classes], y, color=MUTED, marker="o", s=22, zorder=2, edgecolor="white",
               linewidth=0.8, label="COCO val (in-domain)")
    ax.scatter([kitti[c] for c in classes], y, color=INK, marker="s", s=20, zorder=3, edgecolor="white",
               linewidth=0.8, label="KITTI val (shifted)")
    ax.axvline(0, color=AXIS, linewidth=0.6, zorder=0)
    ax.set_yticks(y, classes)
    ax.set_ylim(-0.5, len(classes) - 0.5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("AP(NMS head) - AP(NMS-free head), points")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return _save(fig, out)


def clutter_table(latency_csvs: dict[str, str | Path], key: str = "e2e_ms") -> pd.DataFrame:
    rows = []
    for model, path in latency_csvs.items():
        t = pd.read_csv(path)
        for lo, hi, lbl in CLUTTER_BINS:
            v = t[(t["n_gt"] >= lo) & (t["n_gt"] <= hi)][key]
            if len(v):
                rows.append({"model": model, "bin": lbl, "n": len(v), "p50": v.quantile(0.5), "p99": v.quantile(0.99)})
    return pd.DataFrame(rows)


def fig_latency_vs_clutter(latency_csvs: dict[str, str | Path], out: str | Path, key: str = "e2e_ms") -> Path:
    """F4: median and p99 latency vs objects per frame. Answers what NMS-free buys (RQ2)."""
    style()
    t = clutter_table(latency_csvs, key)
    labels = [b[2] for b in CLUTTER_BINS]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_W, 1.7), sharey=True)
    for ax, stat in zip(axes, ("p50", "p99")):
        for m in latency_csvs:
            sub = t[t["model"] == m].set_index("bin").reindex(labels)
            ax.plot(range(len(labels)), sub[stat], color=MODEL_COLORS[m], marker="o", label=MODEL_LABELS[m])
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel("GT objects per frame")
        ax.set_title(f"{stat} latency", color=INK_2)
    axes[0].set_ylim(bottom=0)          # after both panels, so the p99 panel is not clipped
    axes[0].set_ylabel("end-to-end ms (batch 1)")
    _legend_below(fig, axes[0])
    return _save(fig, out)


def fig_speed_accuracy(points: list[dict], out: str | Path, ylabel: str = "F1 @ IoU 0.5 (KITTI subset)") -> Path:
    """F5: accuracy vs latency on a log axis for every model, VLMs included. Answers RQ3.

    points: [{"model", "latency_ms", "value", "family": closed|open|vlm, "lo"?, "hi"?}]
    """
    style()
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.1))
    for p in points:
        yerr = None if p.get("lo") is None else [[p["value"] - p["lo"]], [p["hi"] - p["value"]]]
        ax.errorbar(p["latency_ms"], p["value"], yerr=yerr, color=MODEL_COLORS[p["model"]],
                    marker=FAMILY_MARKER[p["family"]], markersize=5.5, capsize=0, elinewidth=0.8,
                    linestyle="none", label=MODEL_LABELS[p["model"]])
    # Direct labels only where they cannot collide; clustered points are identified by the legend.
    lx = np.log10([p["latency_ms"] for p in points])
    vy = np.array([p["value"] for p in points])
    for i, p in enumerate(points):
        crowded = any(j != i and abs(lx[i] - lx[j]) < 0.6 and abs(vy[i] - vy[j]) < 0.08 for j in range(len(points)))
        if not crowded:
            ax.annotate(MODEL_LABELS[p["model"]], (p["latency_ms"], p["value"]), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=6.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlim(10 ** np.floor(lx.min()), 10 ** np.ceil(lx.max()))
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("latency per frame, ms (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", ncol=2)
    return _save(fig, out)


def fig_gallery(cases: list[dict], out: str | Path, ncols: int = 2, width: float = PAGE_W) -> Path:
    """F6: crops with GT (white), DontCare (dashed blue), predictions (model color) and the focus box (thick).

    cases: [{"image": path, "crop": [x1, y1, x2, y2], "gt": [[x, y, w, h], ...], "dontcare": [...],
             "preds": [{"model": name, "bbox": [x, y, w, h]}, ...], "focus": [x, y, w, h] | None, "title": str}]
    Panel heights follow each crop's aspect ratio within a row.
    """
    from PIL import Image

    style()
    nrows = int(np.ceil(len(cases) / ncols))
    aspect = [(c["crop"][3] - c["crop"][1]) / (c["crop"][2] - c["crop"][0]) for c in cases]
    row_h = [max(aspect[r * ncols:(r + 1) * ncols]) * width / ncols + 0.22 for r in range(nrows)]
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, sum(row_h)), squeeze=False,
                             gridspec_kw={"height_ratios": row_h})
    for ax in axes.flat:
        ax.axis("off")
    for ax, case in zip(axes.flat, cases):
        x1, y1, x2, y2 = case["crop"]
        ax.imshow(np.asarray(Image.open(case["image"]).convert("RGB"))[int(y1):int(y2), int(x1):int(x2)])
        ax.set_xlim(0, x2 - x1)
        ax.set_ylim(y2 - y1, 0)
        for gx, gy, gw, gh in case.get("dontcare", []):
            ax.add_patch(plt.Rectangle((gx - x1, gy - y1), gw, gh, fill=False, edgecolor="#86b6ef",
                                       linewidth=0.7, linestyle="--"))
        for gx, gy, gw, gh in case.get("gt", []):
            ax.add_patch(plt.Rectangle((gx - x1, gy - y1), gw, gh, fill=False, edgecolor="white", linewidth=0.9))
        for p in case.get("preds", []):
            px, py, pw, ph = p["bbox"]
            ax.add_patch(plt.Rectangle((px - x1, py - y1), pw, ph, fill=False,
                                       edgecolor=MODEL_COLORS.get(p["model"], INK), linewidth=1.0))
        if case.get("focus"):
            fx, fy, fw, fh = case["focus"]
            ax.add_patch(plt.Rectangle((fx - x1 - 3, fy - y1 - 3), fw + 6, fh + 6, fill=False,
                                       edgecolor="#e34948", linewidth=1.8))
        ax.set_title(case.get("title", ""), fontsize=6.5, color=INK_2, loc="left", pad=2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.005, wspace=0.03, hspace=0.12)
    return _save(fig, out)
