from det_eval.finetune import CLASSES, split_train, yolo_lines
from det_eval.kitti import KittiObject


def obj(t, x1, y1, x2, y2):
    return KittiObject(t, 0.0, 0, 0.0, x1, y1, x2, y2)


def test_yolo_lines_maps_and_drops():
    objs = [obj("Car", 100, 50, 300, 150), obj("Van", 0, 0, 10, 10), obj("DontCare", 5, 5, 50, 50),
            obj("Cyclist", 10, 10, 30, 70), obj("Misc", 1, 1, 9, 9)]
    lines = yolo_lines(objs, 1000, 200)
    assert [int(ln.split()[0]) for ln in lines] == [CLASSES.index("car"), CLASSES.index("cyclist")]
    c, cx, cy, w, h = map(float, lines[0].split())
    assert (cx, cy, w, h) == (0.2, 0.5, 0.2, 0.5)


def test_yolo_lines_clips_to_image():
    (line,) = yolo_lines([obj("Pedestrian", -20, 10, 40, 250)], 100, 200)
    _, cx, cy, w, h = map(float, line.split())
    assert abs(cx - 0.2) < 1e-6 and abs(w - 0.4) < 1e-6 and abs(cy + h / 2 - 1.0) < 1e-6


def test_split_is_disjoint_and_complete():
    ids = [f"{i:06d}" for i in range(1000)]
    ft, mv = split_train(ids, 100, seed=0)
    assert len(mv) == 100 and not set(ft) & set(mv) and sorted(ft + mv) == ids
    assert split_train(ids, 100, seed=0)[1] == mv


def test_masked_image_grays_van_but_keeps_overlapping_car(tmp_path):
    import numpy as np
    from PIL import Image

    from det_eval.finetune import MASK_FILL, masked_image

    src = tmp_path / "x.png"
    Image.new("RGB", (100, 50), (10, 200, 30)).save(src)
    objs = [obj("Van", 10, 10, 60, 40), obj("Car", 40, 20, 80, 45), obj("DontCare", 85, 5, 95, 15)]
    im, n = masked_image(src, objs, {"Van"})
    a = np.asarray(im)
    assert n == 1
    assert tuple(a[15, 20]) == MASK_FILL                 # van-only pixel grayed
    assert tuple(a[30, 50]) == (10, 200, 30)             # van/car overlap restored
    assert tuple(a[10, 90]) == (10, 200, 30)             # DontCare not masked unless asked
