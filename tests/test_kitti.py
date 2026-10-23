from collections import Counter

from det_eval.kitti import KittiObject, build_coco_gt, difficulty_level, make_subset, parse_label_line


def _obj(h, occ=0, trunc=0.0):
    return KittiObject("Car", trunc, occ, 0.0, 0.0, 0.0, 10.0, float(h))


def test_parse_label_line():
    o = parse_label_line("Pedestrian 0.00 1 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01")
    assert o.type == "Pedestrian" and o.occluded == 1
    assert abs(o.height - 164.92) < 1e-6


def test_difficulty_levels_match_devkit_definitions():
    assert difficulty_level(_obj(40)) == 0                 # easy: >= 40 px, visible, trunc <= 0.15
    assert difficulty_level(_obj(39)) == 1                 # moderate: >= 25 px
    assert difficulty_level(_obj(60, occ=1)) == 1
    assert difficulty_level(_obj(60, trunc=0.3)) == 1
    assert difficulty_level(_obj(60, occ=2)) == 2          # hard: occlusion up to 2
    assert difficulty_level(_obj(60, trunc=0.5)) == 2
    assert difficulty_level(_obj(24)) == -1                # too small for any level
    assert difficulty_level(_obj(60, occ=3)) == -1         # unknown occlusion
    assert difficulty_level(_obj(60, trunc=0.6)) == -1


def test_closed_set_gt_applies_mapping_and_ignores(kitti_root, mapping):
    gt = build_coco_gt(kitti_root, ["000001", "000002"], mapping)
    assert {c["id"]: c["name"] for c in gt["categories"]} == {1: "person", 3: "car", 7: "train", 8: "truck"}
    assert [im["width"] for im in gt["images"]] == [1242, 1242]

    kinds = Counter((a["kitti_type"], a["category_id"], a["iscrowd"], a["ignore_iou"]) for a in gt["annotations"])
    assert kinds[("Car", 3, 0, 0)] == 3
    assert kinds[("Pedestrian", 1, 0, 0)] == 1
    assert kinds[("Truck", 8, 0, 0)] == 1
    assert kinds[("Van", 3, 0, 1)] == 1                    # neighbor ignore, IoU-matched
    assert kinds[("Person_sitting", 1, 0, 1)] == 1
    assert kinds[("Cyclist", 1, 1, 0)] == 1                # crowd ignore for person
    assert sum(v for k, v in kinds.items() if k[0] == "DontCare") == 4   # one crowd region per category
    assert not any(a["kitti_type"] == "Misc" for a in gt["annotations"])
    assert len(gt["annotations"]) == 12

    levels = {(a["image_id"], a["bbox"][0]): a["level"] for a in gt["annotations"] if a["kitti_type"] == "Car"}
    assert levels == {(1, 100.0): 0, (2, 100.0): 2, (2, 500.0): -1}


def test_open_vocab_gt_adds_cyclist(kitti_root, mapping):
    gt = build_coco_gt(kitti_root, ["000001", "000002"], mapping, open_vocab=True)
    assert 1001 in {c["id"] for c in gt["categories"]}
    cyc = [a for a in gt["annotations"] if a["kitti_type"] == "Cyclist"]
    assert {(a["category_id"], a["iscrowd"]) for a in cyc} == {(1001, 0), (1, 1)}
    assert sum(1 for a in gt["annotations"] if a["kitti_type"] == "DontCare") == 5
    assert len(gt["annotations"]) == 14


def test_copy_frames_reads_straight_from_the_official_zips(kitti_root, tmp_path):
    import zipfile

    from det_eval.kitti import copy_frames

    img_zip, lbl_zip = tmp_path / "images.zip", tmp_path / "labels.zip"
    with zipfile.ZipFile(img_zip, "w") as zi, zipfile.ZipFile(lbl_zip, "w") as zl:
        for sid in ("000001", "000002"):
            zi.write(kitti_root / "image_2" / f"{sid}.png", f"training/image_2/{sid}.png")
            zl.write(kitti_root / "label_2" / f"{sid}.txt", f"training/label_2/{sid}.txt")
    dst = tmp_path / "out"
    assert copy_frames(["000002"], dst, image_zip=img_zip, label_zip=lbl_zip) == 1
    assert sorted(p.name for p in (dst / "image_2").iterdir()) == ["000002.png"]
    assert (dst / "label_2" / "000002.txt").read_text() == (kitti_root / "label_2" / "000002.txt").read_text()


def test_subset_is_deterministic_and_sorted():
    ids = [f"{i:06d}" for i in range(1000)]
    a, b = make_subset(ids, 50, seed=0), make_subset(ids, 50, seed=0)
    assert a == b == sorted(a) and len(set(a)) == 50
    assert make_subset(ids, 50, seed=1) != a
