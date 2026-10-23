import pytest

from det_eval.vlm_grounding import normalize_label, parse_boxes, to_pixels

W, H = 1242, 375
LABELS = {"car": "car", "pedestrian": "person", "person": "person", "cyclist": "cyclist", "tram": "train"}


def test_qwen_style_fenced_json_on_0_1000_scale():
    text = '```json\n[{"bbox_2d": [100, 200, 300, 800], "label": "car"}]\n```'
    boxes, status = parse_boxes(text, W, H)
    assert status == "ok"
    assert boxes[0]["bbox_xyxy"] == pytest.approx([124.2, 75.0, 372.6, 300.0])
    assert boxes[0]["label"] == "car"


def test_reasoning_block_is_stripped():
    text = '<think>the cars are on the left</think>\n[{"bbox_2d": [0, 0, 500, 500], "label": "car"}]'
    boxes, status = parse_boxes(text, W, H)
    assert status == "ok" and len(boxes) == 1


def test_unclosed_think_tag_from_template():
    text = 'reasoning without the opening tag</think>[{"bbox_2d": [0, 0, 10, 10], "label": "tram"}]'
    assert parse_boxes(text, W, H)[1] == "ok"


def test_python_literal_with_single_quotes():
    boxes, status = parse_boxes("[{'bbox_2d': [10, 10, 20, 20], 'label': 'cyclist'}]", W, H)
    assert status == "ok" and boxes[0]["label"] == "cyclist"


def test_truncated_json_is_salvaged():
    text = '[{"bbox_2d": [10, 10, 20, 20], "label": "car"}, {"bbox_2d": [30, 30, 40, 40], "label": "car"}, {"bbox_2d": [50, 5'
    boxes, status = parse_boxes(text, W, H)
    assert status == "salvaged" and len(boxes) == 2


def test_wrapper_dict_and_bare_box_lists():
    boxes, status = parse_boxes('{"car": [[10, 10, 20, 20], [30, 30, 40, 40]], "pedestrian": []}', W, H)
    assert status == "ok" and [b["label"] for b in boxes] == ["car", "car"]


def test_inverted_and_degenerate_boxes():
    boxes, _ = parse_boxes('[{"bbox_2d": [300, 800, 100, 200], "label": "car"}, {"bbox_2d": [5, 5, 5, 9], "label": "car"}]', W, H)
    assert len(boxes) == 1
    assert boxes[0]["bbox_xyxy"] == pytest.approx([124.2, 75.0, 372.6, 300.0])


def test_empty_and_unparseable_outputs():
    assert parse_boxes("[]", W, H) == ([], "empty")
    assert parse_boxes("I cannot see any objects.", W, H) == ([], "fail")


def test_coordinate_modes():
    assert to_pixels([0.1, 0.2, 0.3, 0.8], W, H, "auto") == pytest.approx([124.2, 75.0, 372.6, 300.0])
    assert to_pixels([10, 20, 30, 40], W, H, "pixel") == [10.0, 20.0, 30.0, 40.0]
    assert to_pixels([0, 0, 1000, 1000], W, H, "norm1000") == [0.0, 0.0, 1242.0, 375.0]


def test_label_normalization():
    assert normalize_label("Pedestrians", LABELS) == "person"
    assert normalize_label("  car ", LABELS) == "car"
    assert normalize_label("tram", LABELS) == "train"
    assert normalize_label("traffic light", LABELS) is None
    assert normalize_label(None, LABELS) is None
