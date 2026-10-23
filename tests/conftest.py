"""Synthetic KITTI frames whose labels exercise every mapping and ignore rule."""

from pathlib import Path

import pytest
from PIL import Image

from det_eval.common import load_yaml

REPO = Path(__file__).resolve().parents[1]

# (type, truncated, occluded, x1, y1, x2, y2)
FRAMES = {
    "000001": [
        ("Car", 0.0, 0, 100, 150, 300, 300),          # easy car
        ("Van", 0.0, 0, 400, 140, 600, 300),          # neighbor class of Car -> ignore (IoU)
        ("Pedestrian", 0.0, 0, 700, 120, 740, 220),   # easy pedestrian
        ("Cyclist", 0.0, 0, 800, 130, 860, 250),      # ignore for person (crowd); real GT for open-vocab cyclist
        ("DontCare", -1, -1, 1000, 160, 1100, 200),   # ignore region for every class (crowd)
        ("Misc", 0.0, 0, 50, 50, 90, 90),             # dropped
    ],
    "000002": [
        ("Car", 0.0, 2, 100, 150, 200, 200),          # heavily occluded -> hard only
        ("Car", 0.0, 0, 500, 180, 530, 200),          # 20 px tall -> no difficulty level
        ("Person_sitting", 0.0, 0, 700, 150, 730, 200),  # neighbor class of Pedestrian -> ignore (IoU)
        ("Truck", 0.0, 0, 900, 100, 1200, 300),       # easy truck
    ],
}


def _label_line(t, trunc, occ, x1, y1, x2, y2):
    return f"{t} {trunc:.2f} {occ} -1.57 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} 1.5 1.6 3.9 1.0 1.0 10.0 -1.5"


@pytest.fixture()
def kitti_root(tmp_path):
    (tmp_path / "image_2").mkdir()
    (tmp_path / "label_2").mkdir()
    for sid, objs in FRAMES.items():
        Image.new("RGB", (1242, 375), (90, 90, 90)).save(tmp_path / "image_2" / f"{sid}.png")
        (tmp_path / "label_2" / f"{sid}.txt").write_text("\n".join(_label_line(*o) for o in objs) + "\n")
    return tmp_path


@pytest.fixture()
def mapping():
    return load_yaml(REPO / "configs" / "kitti_to_coco.yaml")
