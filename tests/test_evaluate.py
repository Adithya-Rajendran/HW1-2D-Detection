import copy

import numpy as np
import pytest

from det_eval.evaluate import (
    Bootstrap,
    ap,
    ar,
    coco_from_dict,
    compose_cyclists,
    evaluate_predictions,
    gt_for_mode,
    load_dets,
    operating_point,
    run_cocoeval,
)
from det_eval.kitti import build_coco_gt


def det(img, cat, x1, y1, x2, y2, score):
    return {"image_id": img, "category_id": cat, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score}


PERFECT = [
    det(1, 3, 100, 150, 300, 300, 0.90),   # car A
    det(1, 1, 700, 120, 740, 220, 0.90),   # pedestrian
    det(2, 3, 100, 150, 200, 200, 0.85),   # occluded car B
    det(2, 3, 500, 180, 530, 200, 0.80),   # 20 px car C
    det(2, 8, 900, 100, 1200, 300, 0.90),  # truck
]


@pytest.fixture()
def gt(kitti_root, mapping):
    return build_coco_gt(kitti_root, ["000001", "000002"], mapping)


def coco_ap(gt_dict, dets, **kw):
    g = coco_from_dict(gt_for_mode(gt_dict, "coco"))
    return run_cocoeval(g, load_dets(g, dets), **kw)


def test_perfect_detections_score_one(gt):
    E = coco_ap(gt, PERFECT)
    assert ap(E) == pytest.approx(1.0)


def test_detection_on_van_is_ignored_like_the_devkit(gt):
    on_van = det(1, 3, 400, 140, 600, 300, 0.99)            # top-ranked car detection on a Van
    assert ap(coco_ap(gt, PERFECT + [on_van])) == pytest.approx(1.0)

    no_neighbors = copy.deepcopy(gt)                         # without the rule it would be a false positive
    no_neighbors["annotations"] = [a for a in gt["annotations"] if a["kitti_type"] != "Van"]
    assert ap(coco_ap(no_neighbors, PERFECT + [on_van])) < 0.99


def test_detection_inside_dontcare_is_ignored(gt):
    in_dc = det(1, 3, 1010, 165, 1090, 198, 0.99)
    assert ap(coco_ap(gt, PERFECT + [in_dc])) == pytest.approx(1.0)


def test_rider_box_inside_cyclist_is_not_a_person_false_positive(gt):
    rider = det(1, 1, 800, 130, 860, 200, 0.99)             # COCO 'person' on the rider only
    assert ap(coco_ap(gt, PERFECT + [rider])) == pytest.approx(1.0)

    iou_rule = copy.deepcopy(gt)                             # with IoU matching the rider (IoU 0.58) would count as FP
    for a in iou_rule["annotations"]:
        if a["kitti_type"] == "Cyclist":
            a["iscrowd"], a["ignore_iou"] = 0, 1
    assert ap(coco_ap(iou_rule, PERFECT + [rider])) < 0.99


def test_operating_point_counts(gt):
    dets = [
        det(1, 3, 100, 150, 300, 300, 0.9),                  # TP
        det(1, 3, 1150, 300, 1240, 370, 0.8),                # FP
        det(1, 3, 400, 140, 600, 300, 0.7),                  # on Van: ignored
    ]
    E = coco_ap(gt, dets, cat_ids=[3])
    op = operating_point(E, iou=0.5)
    assert (op["tp"], op["fp"], op["n_gt"]) == (1, 1, 3)
    assert op["precision"] == pytest.approx(0.5) and op["recall"] == pytest.approx(1 / 3)
    assert operating_point(E, iou=0.5, score_thr=0.85)["fp"] == 0


def test_kitti_difficulty_ignores_objects_outside_the_level(gt):
    res = evaluate_predictions(gt, [PERFECT[0]], cat_names=["car"], bootstrap=0)
    kitti = res["kitti"]["AP_R40"]
    assert kitti["easy"]["car"] == pytest.approx(1.0)
    assert kitti["moderate"]["car"] == pytest.approx(1.0)   # occluded car B is ignored at moderate
    assert kitti["hard"]["car"] == pytest.approx(0.5)       # ... and missed at hard (R40: 20 of 40 positions)


def test_height_bins_isolate_small_objects(gt):
    dets = [d for d in PERFECT if d["bbox"][3] >= 25]        # miss only the 20 px car
    res = evaluate_predictions(gt, dets, bootstrap=0)
    bins = res["height_bins"]
    assert bins["<25"]["AP"] == pytest.approx(0.0)
    assert bins["40-80"]["AP"] == pytest.approx(1.0)
    assert bins[">=80"]["AP"] == pytest.approx(1.0)
    assert (bins["<25"]["n_gt"], bins["40-80"]["n_gt"], bins[">=80"]["n_gt"]) == (1, 1, 3)


def _random_coco(seed=0, n_img=40, n_cat=3):
    rng = np.random.default_rng(seed)
    images = [{"id": i + 1, "width": 640, "height": 480, "file_name": f"{i}.jpg"} for i in range(n_img)]
    cats = [{"id": c + 1, "name": f"c{c}"} for c in range(n_cat)]
    anns, dets = [], []
    for im in images:
        for _ in range(rng.integers(0, 6)):
            x, y = rng.uniform(0, 500), rng.uniform(0, 350)
            w, h = rng.uniform(10, 140), rng.uniform(10, 130)
            c = int(rng.integers(1, n_cat + 1))
            anns.append({"id": len(anns) + 1, "image_id": im["id"], "category_id": c, "bbox": [x, y, w, h],
                         "area": w * h, "iscrowd": 0})
            if rng.random() < 0.8:                               # jittered true positive
                j = rng.normal(0, 6, 4)
                dets.append({"image_id": im["id"], "category_id": c, "bbox": [x + j[0], y + j[1], w + j[2], h + j[3]],
                             "score": float(rng.uniform(0.3, 1))})
        for _ in range(rng.integers(0, 4)):                      # false positives
            dets.append({"image_id": im["id"], "category_id": int(rng.integers(1, n_cat + 1)),
                         "bbox": [rng.uniform(0, 500), rng.uniform(0, 350), 40, 40], "score": float(rng.uniform(0, 0.8))})
    return {"images": images, "annotations": anns, "categories": cats}, dets


def test_bootstrap_with_unit_weights_reproduces_pycocotools():
    gt, dets = _random_coco()
    g = coco_from_dict(gt)
    E = run_cocoeval(g, load_dets(g, dets))
    bs = Bootstrap(E)
    assert bs.ap() == pytest.approx(ap(E), abs=1e-9)
    assert bs.ap(iou=0.5) == pytest.approx(ap(E, iou=0.5), abs=1e-9)
    lo, hi = bs.interval(n=200, seed=0)
    assert lo <= ap(E) <= hi


def test_compose_cyclists_pairs_rider_and_bike():
    dets = [det(1, 1, 800, 130, 860, 200, 0.81), det(1, 2, 795, 180, 865, 250, 0.64), det(1, 1, 10, 10, 40, 90, 0.9)]
    out = compose_cyclists(dets)
    assert len(out) == 1
    assert out[0]["bbox"] == [795, 130, 70, 120]
    assert out[0]["score"] == pytest.approx(0.72)


def test_bootstrap_ar_reproduces_pycocotools():
    gt, dets = _random_coco(seed=1)
    g = coco_from_dict(gt)
    E = run_cocoeval(g, load_dets(g, dets))
    assert Bootstrap(E).ar() == pytest.approx(ar(E), abs=1e-9)


def test_paired_bootstrap_identical_models_have_zero_difference():
    from det_eval.compare import paired

    gt, dets = _random_coco(seed=2)
    res = paired(gt, dets, dets, n=50)
    for m in res["groups"]["all"].values():
        assert m["diff"] == 0 and m["diff_ci95"] == [0.0, 0.0] and not m["significant"]


def test_paired_bootstrap_detects_a_clear_gap():
    from det_eval.compare import paired

    gt, dets = _random_coco(seed=3, n_img=80)
    worse = [d for i, d in enumerate(dets) if i % 2]                 # drop half of the boxes
    v = paired(gt, dets, worse, n=200)["groups"]["all"]["AR"]
    assert v["diff"] > 0 and v["significant"]
