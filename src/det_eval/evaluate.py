"""One evaluator for every model (pycocotools underneath).

What it reports for a predictions file:
  coco          AP@[.5:.95], AP50, AP75, AP small/medium/large, AR1/10/100, per-class AP,
                bootstrap 95% intervals, and fixed operating points (P/R/F1 at a score threshold)
  height_bins   AP by box height in pixels (KITTI only): <25, 25-40, 40-80, >=80
  kitti         KITTI-style AP|R40 per class at the devkit IoU (car 0.7, pedestrian/cyclist 0.5)
                for easy/moderate/hard, with devkit ignore semantics
  score-less predictions (VLMs) get operating points instead of AP

KITTI ignore semantics (see configs/kitti_to_coco.yaml):
  * DontCare and Cyclist-for-person are `iscrowd` regions: matched by intersection over detection area.
  * Van-for-car and Person_sitting-for-person are `ignore_iou` GTs. pycocotools ignores a GT whose
    `area` is outside the evaluated range but still matches it with standard IoU, and detections
    matched to it are ignored. Setting area = -1 reproduces the devkit's neighboring-class rule.

Usage:
    python -m det_eval.evaluate --gt GT.json --dets DETS.json --out results/metrics/NAME.json \
        [--img-ids configs/splits/kitti_val_vlm300.txt] [--cats car person] [--bootstrap 500]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .common import load_json, read_ids, save_json, write_manifest

HEIGHT_BINS = ((0.0, 25.0), (25.0, 40.0), (40.0, 80.0), (80.0, 1e5))
HEIGHT_LABELS = ("<25", "25-40", "40-80", ">=80")
KITTI_LEVELS = ("easy", "moderate", "hard")
KITTI_MIN_HEIGHT = (40.0, 25.0, 25.0)                       # devkit: detections below this are ignored
KITTI_IOU = {"car": 0.7, "truck": 0.7, "train": 0.7, "person": 0.5, "cyclist": 0.5}
R40 = np.linspace(1 / 40, 1.0, 40)                          # KITTI's 40 recall positions


# --------------------------------------------------------------------------- pycocotools plumbing

def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def coco_from_dict(dataset: dict) -> COCO:
    coco = COCO()
    coco.dataset = dataset
    _quiet(coco.createIndex)
    return coco


def is_kitti_gt(gt: dict) -> bool:
    return any("ignore_iou" in a for a in gt["annotations"][:2000])


def gt_for_mode(gt: dict, mode: str, level: int | None = None) -> dict:
    """Copy of a KITTI-derived GT dict with `area` rewritten for the evaluation mode.

    coco:   valid GT area = w*h      (COCO small/medium/large by box area)
    height: valid GT area = h**2     (area ranges become box-height ranges)
    kitti:  valid GT area = 1 if the object qualifies for `level`, else -1 (ignored, IoU-matched)
    `ignore_iou` GTs always get area -1. Crowd regions are left alone (pycocotools ignores them).
    """
    out = copy.deepcopy(gt)
    for a in out["annotations"]:
        if a.get("iscrowd", 0):
            continue
        ignored = bool(a.get("ignore_iou", 0))
        w, h = a["bbox"][2], a["bbox"][3]
        if mode == "coco":
            a["area"] = -1.0 if ignored else float(w * h)
        elif mode == "height":
            a["area"] = -1.0 if ignored else float(h) ** 2
        elif mode == "kitti":
            lv = a.get("level", -1)
            a["area"] = 1.0 if (not ignored and 0 <= lv <= level) else -1.0
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return out


def load_dets(coco_gt: COCO, dets: list[dict], mode: str = "coco", cat_ids=None,
              min_height: float | None = None) -> COCO | None:
    """pycocotools result object restricted to GT images/categories; None if nothing is left."""
    img_ids = set(coco_gt.getImgIds())
    cat_ids = set(cat_ids) if cat_ids is not None else None
    keep = [
        dict(d) for d in dets
        if d["image_id"] in img_ids
        and (cat_ids is None or d["category_id"] in cat_ids)
        and (min_height is None or d["bbox"][3] >= min_height)
    ]
    if not keep:
        return None
    coco_dt = _quiet(coco_gt.loadRes, keep)
    if mode == "height":
        for a in coco_dt.dataset["annotations"]:
            a["area"] = float(a["bbox"][3]) ** 2
    return coco_dt


def run_cocoeval(coco_gt: COCO, coco_dt: COCO | None, cat_ids=None, img_ids=None, iou_thrs=None,
                 rec_thrs=None, area_rng=None, area_lbl=None, max_dets=(1, 10, 100)) -> COCOeval | None:
    if coco_dt is None:
        return None
    E = COCOeval(coco_gt, coco_dt, "bbox")
    p = E.params
    if cat_ids is not None:
        p.catIds = sorted(cat_ids)
    if img_ids is not None:
        p.imgIds = sorted(img_ids)
    if iou_thrs is not None:
        p.iouThrs = np.asarray(iou_thrs, dtype=float)
    if rec_thrs is not None:
        p.recThrs = np.asarray(rec_thrs, dtype=float)
    if area_rng is not None:
        p.areaRng = [list(r) for r in area_rng]
        p.areaRngLbl = list(area_lbl)
    p.maxDets = list(max_dets)
    _quiet(E.evaluate)
    _quiet(E.accumulate)
    return E


# --------------------------------------------------------------------------- metrics from COCOeval

def _mean_valid(x) -> float:
    x = np.asarray(x, dtype=float)
    v = x[x > -1]
    return float(v.mean()) if v.size else float("nan")


def _iou_index(E: COCOeval, iou: float) -> int:
    t = np.where(np.isclose(E.params.iouThrs, iou))[0]
    if t.size == 0:
        raise ValueError(f"IoU {iou} was not evaluated (have {E.params.iouThrs})")
    return int(t[0])


def ap(E: COCOeval, iou: float | None = None, area: int = 0, maxdet: int = -1, cat_index: int | None = None) -> float:
    """Mean precision over recall thresholds (and IoUs if iou is None), as COCO's summarize() does."""
    pr = E.eval["precision"]                                   # [T, R, K, A, M]
    if iou is not None:
        pr = pr[[_iou_index(E, iou)]]
    pr = pr[:, :, :, area, maxdet]
    if cat_index is not None:
        pr = pr[:, :, cat_index]
    return _mean_valid(pr)


def ar(E: COCOeval, area: int = 0, maxdet: int = -1, cat_index: int | None = None) -> float:
    rc = E.eval["recall"][:, :, area, maxdet]                  # [T, K]
    if cat_index is not None:
        rc = rc[:, cat_index]
    return _mean_valid(rc)


def _eval_img(E: COCOeval, k: int, area: int, i: int):
    p = E.params
    n_area, n_img = len(p.areaRng), len(p.imgIds)
    return E.evalImgs[k * n_area * n_img + area * n_img + i]


def _ranked(E: COCOeval, t: int, area: int = 0, maxdet: int = -1, cat_indices=None):
    """Scores, TP flags and FP flags of all non-ignored detections, ranked by score; plus #valid GT."""
    p = E.params
    md = p.maxDets[maxdet]
    ks = range(len(p.catIds)) if cat_indices is None else cat_indices
    scores, tps, fps, n_gt = [], [], [], 0
    for k in ks:
        for i in range(len(p.imgIds)):
            e = _eval_img(E, k, area, i)
            if e is None:
                continue
            n_gt += int((~np.asarray(e["gtIgnore"], dtype=bool)).sum())
            s = np.asarray(e["dtScores"][:md], dtype=float)
            if not s.size:
                continue
            dm = np.asarray(e["dtMatches"])[t, :md] > 0
            di = np.asarray(e["dtIgnore"])[t, :md].astype(bool)
            scores.append(s[~di])
            tps.append(dm[~di])
            fps.append(~dm[~di])
    if not scores:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0, bool), n_gt
    s = np.concatenate(scores)
    order = np.argsort(-s, kind="mergesort")
    return s[order], np.concatenate(tps)[order], np.concatenate(fps)[order], n_gt


def operating_point(E: COCOeval, iou: float = 0.5, score_thr: float | None = None, area: int = 0,
                    maxdet: int = -1, cat_indices=None) -> dict:
    """TP/FP/FN, precision, recall and F1 using every detection with score >= score_thr (all if None).

    Uses the same greedy matches pycocotools made for AP. Dropping low-score detections does not
    change how higher-score detections were matched, so the counts are exact.
    """
    s, tp, fp, n_gt = _ranked(E, _iou_index(E, iou), area, maxdet, cat_indices)
    keep = np.ones_like(s, dtype=bool) if score_thr is None else s >= score_thr
    n_tp, n_fp = int(tp[keep].sum()), int(fp[keep].sum())
    prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else float("nan")
    rec = n_tp / n_gt if n_gt else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if n_tp else 0.0
    return {"iou": iou, "score_thr": score_thr, "tp": n_tp, "fp": n_fp, "fn": n_gt - n_tp, "n_gt": n_gt,
            "precision": prec, "recall": rec, "f1": f1}


def precision_at_recall(E: COCOeval, target: float, iou: float = 0.5, area: int = 0, maxdet: int = -1,
                        cat_indices=None) -> dict:
    """Precision at the first rank where recall reaches `target` (non-interpolated), pooled over classes.

    Used to compare scored detectors with a VLM that has a single operating point.
    """
    s, tp, fp, n_gt = _ranked(E, _iou_index(E, iou), area, maxdet, cat_indices)
    if not n_gt or not s.size:
        return {"target_recall": target, "precision": float("nan"), "score_thr": float("nan"), "reached": False}
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rc = tp_c / n_gt
    idx = int(np.searchsorted(rc, target, side="left"))
    if idx >= rc.size:
        return {"target_recall": target, "precision": float("nan"), "score_thr": float("nan"),
                "reached": False, "max_recall": float(rc[-1])}
    return {"target_recall": target, "precision": float(tp_c[idx] / (tp_c[idx] + fp_c[idx])),
            "recall": float(rc[idx]), "score_thr": float(s[idx]), "reached": True}


def best_f1(E: COCOeval, iou: float = 0.5, area: int = 0, maxdet: int = -1, cat_indices=None) -> dict:
    """Highest F1 over all score thresholds, pooled over classes."""
    s, tp, fp, n_gt = _ranked(E, _iou_index(E, iou), area, maxdet, cat_indices)
    if not n_gt or not s.size:
        return {"iou": iou, "f1": 0.0, "precision": float("nan"), "recall": 0.0, "score_thr": float("nan")}
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    prec, rec = tp_c / (tp_c + fp_c), tp_c / n_gt
    f1 = np.where(tp_c > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    j = int(np.argmax(f1))
    return {"iou": iou, "f1": float(f1[j]), "precision": float(prec[j]), "recall": float(rec[j]), "score_thr": float(s[j])}


# --------------------------------------------------------------------------- bootstrap

class Bootstrap:
    """Image-level bootstrap of AP from pycocotools' per-image match tables.

    Resampling images with replacement is done with integer weights per image. With all weights 1
    it reproduces COCOeval.accumulate() exactly (tested), so the point estimate and its interval
    come from the same computation.
    """

    def __init__(self, E: COCOeval, area: int = 0, maxdet: int = -1):
        p = E.params
        self.n_img = len(p.imgIds)
        self.n_iou = len(p.iouThrs)
        self.rec_thrs = np.asarray(p.recThrs, dtype=float)
        self.iou_thrs = np.asarray(p.iouThrs, dtype=float)
        md = p.maxDets[maxdet]
        self.per_cat = []
        for k in range(len(p.catIds)):
            scores, img_idx, tps, fps = [], [], [], []
            n_valid = np.zeros(self.n_img)
            for i in range(self.n_img):
                e = _eval_img(E, k, area, i)
                if e is None:
                    continue
                n_valid[i] = (~np.asarray(e["gtIgnore"], dtype=bool)).sum()
                s = np.asarray(e["dtScores"][:md], dtype=float)
                if not s.size:
                    continue
                dm = np.asarray(e["dtMatches"])[:, :md] > 0
                di = np.asarray(e["dtIgnore"])[:, :md].astype(bool)
                scores.append(s)
                img_idx.append(np.full(s.size, i))
                tps.append(dm & ~di)
                fps.append(~dm & ~di)
            if scores:
                order = np.argsort(-np.concatenate(scores), kind="mergesort")
                entry = (np.concatenate(img_idx)[order], np.concatenate(tps, axis=1)[:, order],
                         np.concatenate(fps, axis=1)[:, order], n_valid)
            else:
                entry = (np.zeros(0, dtype=int), np.zeros((self.n_iou, 0), bool), np.zeros((self.n_iou, 0), bool), n_valid)
            self.per_cat.append(entry)

    def _ap_t(self, entry, w: np.ndarray) -> np.ndarray:
        """AP at each IoU threshold for one category under image weights w."""
        img_idx, tp, fp, n_valid = entry
        n_gt = float((w * n_valid).sum())
        if n_gt == 0:
            return np.full(self.n_iou, np.nan)
        wd = w[img_idx]
        keep = wd > 0
        if not keep.any():
            return np.zeros(self.n_iou)
        wd, tp, fp = wd[keep], tp[:, keep], fp[:, keep]
        tp_sum = np.cumsum(tp * wd, axis=1)
        fp_sum = np.cumsum(fp * wd, axis=1)
        out = np.zeros(self.n_iou)
        for t in range(self.n_iou):
            rc = tp_sum[t] / n_gt
            pr = tp_sum[t] / (fp_sum[t] + tp_sum[t] + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1]          # precision envelope
            inds = np.searchsorted(rc, self.rec_thrs, side="left")
            q = np.where(inds < pr.size, pr[np.minimum(inds, pr.size - 1)], 0.0)
            out[t] = q.mean()
        return out

    def _t_mask(self, iou: float | None) -> np.ndarray:
        if iou is None:
            return np.ones(self.n_iou, dtype=bool)
        return np.isclose(self.iou_thrs, iou)

    def per_class(self, weights: np.ndarray | None = None, iou: float | None = None) -> np.ndarray:
        w = np.ones(self.n_img) if weights is None else weights
        mask = self._t_mask(iou)
        return np.array([self._ap_t(e, w)[mask].mean() for e in self.per_cat])

    def ap(self, weights: np.ndarray | None = None, iou: float | None = None) -> float:
        v = self.per_class(weights, iou)
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else float("nan")

    def ar(self, weights: np.ndarray | None = None, iou: float | None = None) -> float:
        """Recall with all kept detections (COCO AR at this maxDets), mean over IoU thresholds and classes."""
        w = np.ones(self.n_img) if weights is None else weights
        mask = self._t_mask(iou)
        vals = []
        for img_idx, tp, _, n_valid in self.per_cat:
            n_gt = float((w * n_valid).sum())
            if n_gt == 0:
                continue
            vals.append(((tp * w[img_idx]).sum(axis=1) / n_gt)[mask].mean() if img_idx.size else 0.0)
        return float(np.mean(vals)) if vals else float("nan")

    def interval(self, n: int = 500, seed: int = 0, alpha: float = 0.05, iou: float | None = None) -> list[float]:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n):
            w = np.bincount(rng.integers(0, self.n_img, self.n_img), minlength=self.n_img).astype(float)
            vals.append(self.ap(w, iou))
        vals = np.asarray(vals)
        vals = vals[~np.isnan(vals)]
        if not vals.size:
            return [float("nan"), float("nan")]
        return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]


# --------------------------------------------------------------------------- closed-set cyclist baseline

def compose_cyclists(dets: list[dict], person_id: int = 1, bicycle_id: int = 2, cyclist_id: int = 1001,
                     min_overlap: float = 0.2, min_score: float = 0.05) -> list[dict]:
    """Rule-based 'cyclist' boxes from a closed-set COCO detector (COCO has no cyclist class).

    Pair each person with the bicycle that overlaps it most (intersection over person area >= min_overlap,
    horizontal centers within half a bicycle width), greedily by score, one bicycle per person.
    The cyclist box is the union of the pair; its score is the geometric mean of the two scores.
    """
    by_img: dict[int, dict[str, list]] = {}
    for d in dets:
        if d["score"] < min_score or d["category_id"] not in (person_id, bicycle_id):
            continue
        key = "p" if d["category_id"] == person_id else "b"
        by_img.setdefault(d["image_id"], {"p": [], "b": []})[key].append(d)
    out = []
    for img_id, groups in by_img.items():
        used = set()
        for p in sorted(groups["p"], key=lambda d: -d["score"]):
            px, py, pw, ph = p["bbox"]
            best, best_ov = None, min_overlap
            for j, b in enumerate(groups["b"]):
                if j in used:
                    continue
                bx, by, bw, bh = b["bbox"]
                iw = min(px + pw, bx + bw) - max(px, bx)
                ih = min(py + ph, by + bh) - max(py, by)
                if iw <= 0 or ih <= 0 or abs((px + pw / 2) - (bx + bw / 2)) > bw / 2:
                    continue
                ov = iw * ih / max(pw * ph, 1e-9)
                if ov >= best_ov:
                    best, best_ov = j, ov
            if best is None:
                continue
            used.add(best)
            b = groups["b"][best]
            bx, by, bw, bh = b["bbox"]
            x1, y1 = min(px, bx), min(py, by)
            x2, y2 = max(px + pw, bx + bw), max(py + ph, by + bh)
            out.append({"image_id": img_id, "category_id": cyclist_id, "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(np.sqrt(p["score"] * b["score"]))})
    return out


# --------------------------------------------------------------------------- full evaluation

def _summary(E: COCOeval, names: dict[int, str]) -> dict:
    p = E.params
    out = {"AP": ap(E), "AP50": ap(E, iou=0.5), "AP75": ap(E, iou=0.75)}
    for a_idx, lbl in enumerate(p.areaRngLbl[1:], start=1):
        out[f"AP_{lbl}"] = ap(E, area=a_idx)
    for m_idx, md in enumerate(p.maxDets):
        out[f"AR{md}"] = ar(E, maxdet=m_idx)
    out["per_class_AP"] = {names[c]: ap(E, cat_index=k) for k, c in enumerate(p.catIds)}
    out["per_class_AP50"] = {names[c]: ap(E, iou=0.5, cat_index=k) for k, c in enumerate(p.catIds)}
    return out


def evaluate_predictions(gt: dict, dets: list[dict], img_ids=None, cat_names=None, bootstrap: int = 500,
                         conf_deploy: float = 0.25, seed: int = 0, scored: bool | None = None) -> dict:
    """Every metric this project reports, for one predictions list against one GT dict.

    `scored=None` auto-detects: predictions are score-less (VLM output) when every score is exactly 1.0,
    which is what det_eval.vlm_grounding writes. Score-less runs get operating points, not AP.
    """
    names_all = {c["id"]: c["name"] for c in gt["categories"]}
    by_name = {v: k for k, v in names_all.items()}
    cat_ids = sorted(by_name[n] for n in cat_names) if cat_names else sorted(names_all)
    names = {c: names_all[c] for c in cat_ids}
    if img_ids is not None:
        img_ids = sorted(set(img_ids) & {im["id"] for im in gt["images"]})
    kitti = is_kitti_gt(gt)
    if scored is None:
        scored = any(d["score"] != 1.0 for d in dets)
    res = {
        "n_images": len(img_ids) if img_ids is not None else len(gt["images"]),
        "n_dets": len(dets),
        "categories": [names[c] for c in cat_ids],
        "scored": scored,
        "kitti_protocol": kitti,
    }

    # COCO-style metrics and operating points
    coco_gt = coco_from_dict(gt_for_mode(gt, "coco") if kitti else gt)
    E = run_cocoeval(coco_gt, load_dets(coco_gt, dets, "coco", cat_ids), cat_ids, img_ids)
    if E is None:
        res["coco"] = None
        return res
    res["coco"] = _summary(E, names) if scored else {}
    res["operating_points"] = {
        f"iou{iou}": {
            "all_dets": operating_point(E, iou),
            f"conf{conf_deploy}": operating_point(E, iou, conf_deploy) if scored else None,
            "best_f1": best_f1(E, iou) if scored else None,
            "per_class_all_dets": {names[c]: operating_point(E, iou, None, cat_indices=[k]) for k, c in enumerate(E.params.catIds)},
        }
        for iou in (0.5, 0.7)
    }
    if scored and bootstrap:
        bs = Bootstrap(E)
        res["coco"]["AP_ci95"] = bs.interval(bootstrap, seed)
        res["coco"]["AP50_ci95"] = bs.interval(bootstrap, seed, iou=0.5)
        res["coco"]["bootstrap_n"] = bootstrap

    if not kitti:
        return res

    # AP by box height (pixels)
    gt_h = coco_from_dict(gt_for_mode(gt, "height"))
    area_rng = [[0, 1e10]] + [[lo ** 2, hi ** 2 - 1e-6] for lo, hi in HEIGHT_BINS]
    E_h = run_cocoeval(gt_h, load_dets(gt_h, dets, "height", cat_ids), cat_ids, img_ids,
                       area_rng=area_rng, area_lbl=["all", *HEIGHT_LABELS])
    if E_h is not None:
        bins = {}
        img_set = set(img_ids) if img_ids is not None else None
        for a_idx, lbl in enumerate(HEIGHT_LABELS, start=1):
            entry = {"n_gt": int(sum(1 for a in gt_h.dataset["annotations"]
                                     if not a.get("iscrowd") and a["category_id"] in names
                                     and area_rng[a_idx][0] <= a["area"] <= area_rng[a_idx][1]
                                     and (img_set is None or a["image_id"] in img_set)))}
            if scored:
                entry.update({"AP": ap(E_h, area=a_idx), "AP50": ap(E_h, iou=0.5, area=a_idx),
                              "per_class_AP": {names[c]: ap(E_h, area=a_idx, cat_index=k) for k, c in enumerate(E_h.params.catIds)}})
                if bootstrap:
                    entry["AP_ci95"] = Bootstrap(E_h, area=a_idx).interval(max(bootstrap // 2, 100), seed)
            entry["recall_iou0.5_all_dets"] = operating_point(E_h, 0.5, area=a_idx)["recall"]
            bins[lbl] = entry
        res["height_bins"] = bins

    # KITTI-style AP|R40 at the devkit IoU per class and difficulty (only for GT with KITTI difficulty levels)
    if scored and any(a.get("level", -1) >= 0 for a in gt["annotations"]):
        kitti_res = {}
        for level, lname in enumerate(KITTI_LEVELS):
            gt_k = coco_from_dict(gt_for_mode(gt, "kitti", level))
            kitti_res[lname] = {}
            for c in cat_ids:
                thr = KITTI_IOU.get(names[c], 0.5)
                dt = load_dets(gt_k, dets, "coco", [c], min_height=KITTI_MIN_HEIGHT[level])
                E_k = run_cocoeval(gt_k, dt, [c], img_ids, iou_thrs=[thr], rec_thrs=R40,
                                   area_rng=[[0, 1e10]], area_lbl=["all"], max_dets=(100,))
                kitti_res[lname][names[c]] = ap(E_k) if E_k is not None else 0.0
        res["kitti"] = {"iou": {names[c]: KITTI_IOU.get(names[c], 0.5) for c in cat_ids}, "AP_R40": kitti_res}
    return res


def main(argv: list[str] | None = None) -> None:
    ap_ = argparse.ArgumentParser(prog="python -m det_eval.evaluate", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--gt", required=True)
    ap_.add_argument("--dets", required=True)
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--img-ids", help="split file restricting evaluation to these frames")
    ap_.add_argument("--cats", nargs="+", help="category names to evaluate (default: all in GT)")
    ap_.add_argument("--bootstrap", type=int, default=500)
    ap_.add_argument("--conf-deploy", type=float, default=0.25)
    ap_.add_argument("--compose-cyclists", action="store_true",
                     help="add rule-based cyclist boxes from person+bicycle detections (closed-set baseline)")
    args = ap_.parse_args(argv)

    gt = load_json(args.gt)
    dets = load_json(args.dets)
    if args.compose_cyclists:
        dets = dets + compose_cyclists(dets)
    img_ids = [int(i) for i in read_ids(args.img_ids)] if args.img_ids else None
    res = evaluate_predictions(gt, dets, img_ids, args.cats, args.bootstrap, args.conf_deploy)
    res.update({"gt": args.gt, "dets": args.dets, "img_ids_file": args.img_ids})
    save_json(res, args.out, indent=2)
    write_manifest(args.out, gt=args.gt, dets=args.dets, bootstrap=args.bootstrap)
    coco = res.get("coco") or {}
    if coco:
        print(f"{Path(args.dets).name}: AP {coco['AP']:.3f}  AP50 {coco['AP50']:.3f}  AP75 {coco['AP75']:.3f}")
    else:
        op = res["operating_points"]["iou0.5"]["all_dets"]
        print(f"{Path(args.dets).name}: P {op['precision']:.3f}  R {op['recall']:.3f}  F1 {op['f1']:.3f} @IoU0.5")


if __name__ == "__main__":
    main()
