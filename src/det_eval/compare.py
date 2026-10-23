"""Paired image-level bootstrap for the difference between two detectors on the same frames.

Both models are scored by the same evaluator; each bootstrap replicate draws one set of image weights
and applies it to both, so per-image difficulty cancels and the interval is for AP(A) - AP(B) itself.
Two unpaired 95% intervals that overlap do not imply "no difference"; this test answers that question.

Usage:
    python -m det_eval.compare --gt data/kitti/annotations/kitti_val_coco.json \
        --a runs/preds/yolo26m_e2e__kitti_val__640.json --b runs/preds/yolo26m_nms__kitti_val__640.json \
        --out results/paired/e2e_vs_nms__kitti_val__640.json
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import load_json, read_ids, save_json
from .evaluate import (HEIGHT_BINS, HEIGHT_LABELS, Bootstrap, coco_from_dict, gt_for_mode, is_kitti_gt,
                       load_dets, run_cocoeval)


def _evals(gt: dict, dets: list[dict], cat_ids, img_ids, height: bool):
    if height:
        g = coco_from_dict(gt_for_mode(gt, "height"))
        area_rng = [[0, 1e10]] + [[lo ** 2, hi ** 2 - 1e-6] for lo, hi in HEIGHT_BINS]
        return run_cocoeval(g, load_dets(g, dets, "height", cat_ids), cat_ids, img_ids,
                            area_rng=area_rng, area_lbl=["all", *HEIGHT_LABELS])
    g = coco_from_dict(gt_for_mode(gt, "coco") if is_kitti_gt(gt) else gt)
    return run_cocoeval(g, load_dets(g, dets, "coco", cat_ids), cat_ids, img_ids)


def _stats(name, a_vals, b_vals, d_vals, alpha):
    d = np.asarray(d_vals)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return {"metric": name, "a": a_vals, "b": b_vals, "diff": a_vals - b_vals,
            "diff_ci95": [float(lo), float(hi)],
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            "significant": bool(lo > 0 or hi < 0)}


def paired(gt: dict, dets_a: list[dict], dets_b: list[dict], img_ids=None, cat_names=None, n: int = 1000,
           seed: int = 0, alpha: float = 0.05, only_groups=None) -> dict:
    names = {c["id"]: c["name"] for c in gt["categories"]}
    cat_ids = sorted(k for k, v in names.items() if not cat_names or v in cat_names)
    if img_ids is not None:
        img_ids = sorted(set(img_ids) & {im["id"] for im in gt["images"]})
    groups = {"all": (False, 0)}
    if is_kitti_gt(gt):
        groups.update({f"h{lbl}": (True, i) for i, lbl in enumerate(HEIGHT_LABELS, start=1)})
    if only_groups:
        groups = {k: v for k, v in groups.items() if k in only_groups}
    out = {"n_boot": n, "seed": seed, "categories": [names[c] for c in cat_ids], "groups": {}}
    cache = {}
    for g, (height, area) in groups.items():
        if height not in cache:
            cache[height] = (_evals(gt, dets_a, cat_ids, img_ids, height), _evals(gt, dets_b, cat_ids, img_ids, height))
        Ea, Eb = cache[height]
        A, B = Bootstrap(Ea, area=area), Bootstrap(Eb, area=area)
        assert A.n_img == B.n_img
        t50 = int(np.argmin(np.abs(A.iou_thrs - 0.5)))

        def stats(bs, w):
            """AP, AP50, AR, R50 from one pass over the match tables (all share the same image weights)."""
            ap_t = np.array([bs._ap_t(e, np.ones(bs.n_img) if w is None else w) for e in bs.per_cat])
            ok = ~np.isnan(ap_t[:, 0])
            return np.array([ap_t[ok].mean(), ap_t[ok, t50].mean(), bs.ar(w), bs.ar(w, 0.5)])

        point_a, point_b = stats(A, None), stats(B, None)
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(n):
            w = np.bincount(rng.integers(0, A.n_img, A.n_img), minlength=A.n_img).astype(float)
            diffs.append(stats(A, w) - stats(B, w))
        diffs = np.asarray(diffs)
        out["groups"][g] = {m: _stats(m, float(point_a[i]), float(point_b[i]),
                                      diffs[np.isfinite(diffs[:, i]), i], alpha)
                            for i, m in enumerate(("AP", "AP50", "AR", "R50"))}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m det_eval.compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--a", required=True, help="predictions of model A (difference is A - B)")
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-ids", help="split file restricting evaluation to these frames")
    ap.add_argument("--cats", nargs="+")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", nargs="+", help="subset of: all h<25 h25-40 h40-80 h>=80 (default: all of them)")
    args = ap.parse_args(argv)
    ids = [int(i) for i in read_ids(args.img_ids)] if args.img_ids else None
    res = paired(load_json(args.gt), load_json(args.a), load_json(args.b), ids, args.cats, args.n, args.seed, only_groups=args.groups)
    res.update({"gt": args.gt, "a": args.a, "b": args.b})
    save_json(res, args.out, indent=2)
    for g, ms in res["groups"].items():
        print(g, "  ".join(f"{m} {v['diff']*100:+.2f} [{v['diff_ci95'][0]*100:+.2f}, {v['diff_ci95'][1]*100:+.2f}]"
                           f"{'*' if v['significant'] else ''}" for m, v in ms.items()))


if __name__ == "__main__":
    main()
