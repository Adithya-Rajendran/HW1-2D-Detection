import json

from det_eval.kitti import KittiObject
from det_eval.vlm_finetune import SKIP_MODULES, answer_text
from det_eval.vlm_grounding import parse_boxes


def obj(t, x1, y1, x2, y2):
    return KittiObject(t, 0.0, 0, 0.0, x1, y1, x2, y2)


def test_answer_matches_the_base_model_format_and_round_trips():
    txt = answer_text([obj("Truck", 600, 100, 800, 300), obj("Car", 100, 50, 300, 150), obj("Van", 0, 0, 50, 50),
                       obj("Cyclist", 900, 100, 950, 250)], 1000, 500)
    assert txt.startswith("```json\n[\n    {\"bbox_2d\": [100, 100, 300, 300], \"label\": \"car\"}")
    boxes, status = parse_boxes(txt, 1000, 500, "norm1000")
    assert status == "ok" and [b["label"] for b in boxes] == ["car", "truck", "cyclist"]
    assert boxes[0]["bbox_xyxy"] == [100.0, 50.0, 300.0, 150.0]


def test_empty_frame_and_vision_modules_skipped():
    assert json.loads(answer_text([obj("DontCare", 1, 1, 9, 9)], 100, 100).strip("`json\n")) == []
    assert SKIP_MODULES.search("model.vision_tower.blocks.0.attn.q_proj")
    assert not SKIP_MODULES.search("model.language_model.layers.3.self_attn.q_proj")


def test_lora_regex_excludes_vision_tower_and_head():
    import re

    import torch

    from det_eval.vlm_finetune import lora_targets

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(2, 2)})
            self.language_model = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(2, 2), "up_proj": torch.nn.Linear(2, 2)})
            self.lm_head = torch.nn.Linear(2, 2)

    regex, leaves = lora_targets(M())
    assert leaves == ["q_proj", "up_proj"]
    assert [n for n, _ in M().named_modules() if re.fullmatch(regex, n)] == ["language_model.q_proj", "language_model.up_proj"]


def test_compact_format_size_order_and_parse_with_spans():
    from det_eval.vlm_grounding import box_score, parse_boxes

    txt = answer_text([obj("Car", 100, 50, 300, 150), obj("Truck", 600, 100, 800, 300)], 1000, 500, "compact", "size")
    assert txt == "truck 600 200 800 600\ncar 100 100 300 300"
    boxes, status = parse_boxes(txt, 1000, 500, "norm1000")
    assert status == "ok" and [b["label"] for b in boxes] == ["truck", "car"]
    assert boxes[1]["bbox_xyxy"] == [100.0, 50.0, 300.0, 150.0] and boxes[1]["span"] == [22, 41]
    assert parse_boxes("none", 1000, 500)[1] == "empty"
    info = [((0, 5), -0.1), ((5, 21), -0.2), ((21, 22), -5.0), ((22, 41), -1.0)]
    assert abs(box_score(boxes[1]["span"], info) - 2.718281828 ** -1.0) < 1e-6


def test_json_answer_gets_spans():
    from det_eval.vlm_grounding import parse_boxes

    txt = answer_text([obj("Car", 100, 50, 300, 150), obj("Truck", 600, 100, 800, 300)], 1000, 500)
    boxes, _ = parse_boxes(txt, 1000, 500)
    assert all(b["span"] for b in boxes) and txt[boxes[0]["span"][0]] == "{"
