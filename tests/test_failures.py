import pytest

from det_eval.failures import analyze
from det_eval.kitti import build_coco_gt


def det(img, cat, x1, y1, x2, y2, score=0.9):
    return {"image_id": img, "category_id": cat, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score}


@pytest.fixture()
def gt(kitti_root, mapping):
    return build_coco_gt(kitti_root, ["000001", "000002"], mapping)


def test_error_types(gt):
    dets = [
        det(1, 3, 100, 150, 300, 300),            # car A found
        det(1, 3, 105, 150, 300, 300, 0.8),       # duplicate of car A
        det(1, 8, 400, 140, 600, 300),            # truck on the Van: confused
        det(1, 1, 1150, 100, 1160, 115),          # tiny box on nothing: unlabeled_small
        det(1, 1, 700, 120, 740, 220, 0.1),       # pedestrian below the deploy threshold: not kept
        det(2, 3, 100, 150, 140, 200),            # loose box on occluded car B (IoU 0.4)
        det(2, 8, 500, 180, 530, 200),            # 20 px car C labeled truck: confused
        det(2, 8, 900, 100, 1200, 300),           # truck found
    ]
    r = analyze(gt, dets)
    assert r["tp"] == 2
    assert r["fp_types"]["duplicate"] == 1 and r["fp_types"]["confused"] == 2
    assert r["fp_types"]["unlabeled_small"] == 1 and r["fp_types"]["loose"] == 1
    assert r["fn_types"]["missed"] == 1                     # the pedestrian (its box was below 0.25)
    assert r["fn_types"]["loose"] == 1 and r["fn_types"]["confused"] == 1
    assert r["fn"] == 3 and r["fp"] == 5
