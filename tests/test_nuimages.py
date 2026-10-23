from pathlib import Path

from det_eval.common import load_yaml
from det_eval.nuimages import build_coco_gt, make_subset

REPO = Path(__file__).resolve().parents[1]


def tables():
    cats = ["vehicle.car", "vehicle.truck", "vehicle.bus.bendy", "vehicle.bicycle", "human.pedestrian.adult",
            "vehicle.trailer", "movable_object.barrier", "human.pedestrian.stroller"]
    category = [{"token": f"c{i}", "name": n} for i, n in enumerate(cats)]
    attribute = [{"token": "a0", "name": "cycle.with_rider"}, {"token": "a1", "name": "cycle.without_rider"},
                 {"token": "a2", "name": "pedestrian.sitting_lying_down"}]
    sd = [{"token": f"s{i}", "filename": f"samples/CAM_FRONT/{i}.jpg", "width": 1600, "height": 900,
           "is_key_frame": i < 4} for i in range(6)]
    c = {n: f"c{i}" for i, n in enumerate(cats)}

    def oa(s, cat, box, attrs=()):
        return {"token": f"o{s}{cat}", "sample_data_token": s, "category_token": c[cat], "attribute_tokens": list(attrs), "bbox": box}

    obj = [oa("s0", "vehicle.car", [100, 100, 300, 200]), oa("s0", "vehicle.truck", [400, 100, 600, 300]),
           oa("s0", "vehicle.bus.bendy", [700, 100, 900, 300]), oa("s0", "vehicle.bicycle", [1000, 200, 1050, 300], ["a0"]),
           oa("s0", "vehicle.bicycle", [1100, 200, 1150, 300], ["a1"]), oa("s0", "human.pedestrian.adult", [1200, 100, 1240, 250]),
           oa("s0", "vehicle.trailer", [10, 10, 90, 90]), oa("s0", "movable_object.barrier", [5, 500, 50, 520]),
           oa("s0", "human.pedestrian.stroller", [1300, 200, 1330, 260]),
           {**oa("s0", "human.pedestrian.adult", [1400, 200, 1440, 260], ["a2"]), "token": "sitting"}]
    return {"category": category, "attribute": attribute, "sample_data": sd, "object_ann": obj}


def test_subset_uses_key_frames_only():
    t = tables()
    assert make_subset(t, 10) == ["s0", "s1", "s2", "s3"]
    assert len(make_subset(t, 2, seed=1)) == 2


def test_mapping_and_ignores():
    gt = build_coco_gt(tables(), ["s0"], load_yaml(REPO / "configs" / "nuimages_to_coco.yaml"), open_vocab=True)
    names = {c["id"]: c["name"] for c in gt["categories"]}
    real = sorted(names[a["category_id"]] for a in gt["annotations"] if not a["iscrowd"] and not a["ignore_iou"])
    assert real == ["bus", "car", "cyclist", "person", "truck"]
    ign = sorted((a["src_type"], names[a["category_id"]], a["iscrowd"]) for a in gt["annotations"] if a["iscrowd"] or a["ignore_iou"])
    assert ign == [("human.pedestrian.adult", "person", 0), ("human.pedestrian.stroller", "person", 0),
                   ("vehicle.bicycle", "person", 1), ("vehicle.trailer", "truck", 0)]
    assert gt["info"]["unmapped"] == {"vehicle.bicycle": 1, "movable_object.barrier": 1}
