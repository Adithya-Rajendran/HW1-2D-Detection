set -x
cd ~/Projects/SJSU/Fall26/CMPE249/HW-2D-Detection
PY=.venv/bin/python
# Cosmos 3 Edge weights in the background (network only)
(.venv/bin/hf download nvidia/Cosmos3-Edge > runs/logs/hf_cosmos3_edge.log 2>&1 || .venv/bin/huggingface-cli download nvidia/Cosmos3-Edge >> runs/logs/hf_cosmos3_edge.log 2>&1) &
DL=$!
# 1) score everything, report, figures
$PY -m det_eval.run evaluate && $PY -m det_eval.run report && $PY -m det_eval.run figures
# 2) latency: PyTorch, then TensorRT FP16
$PY -m det_eval.run latency
$PY -m det_eval.run latency --engine
# 3) FP32 vs FP16 check on 1,000 COCO images (YOLO26m, NMS head)
$PY - <<PYEOF
import random, json
from det_eval.common import load_json, save_json
from det_eval.predict_ultralytics import run_prediction
from det_eval.evaluate import evaluate_predictions
gt = load_json("data/coco/annotations/instances_val2017.json")
ids = set(sorted(random.Random(0).sample([im["id"] for im in gt["images"]], 1000)))
sub = {**gt, "images": [im for im in gt["images"] if im["id"] in ids],
       "annotations": [a for a in gt["annotations"] if a["image_id"] in ids]}
save_json(sub, "runs/check/coco_val_1000.json")
out = {}
for name, fp16 in (("fp32", False), ("fp16", True)):
    p = run_prediction("yolo", "yolo26m.pt", "runs/check/coco_val_1000.json", "data/coco/images/val2017",
                       f"runs/check/yolo26m_nms_{name}.json", 640, fp16=fp16, name=name)
    r = evaluate_predictions(sub, load_json(p), bootstrap=0)
    out[name] = round(100 * r["coco"]["AP"], 2)
full = evaluate_predictions(sub, load_json("runs/preds/yolo26m_nms__coco_val2017__640.json"), bootstrap=0)
out["fp16_fullrun_on_subset"] = round(100 * full["coco"]["AP"], 2)
print("FP_CHECK", json.dumps(out)); save_json(out, "results/fp32_vs_fp16_yolo26m.json", indent=2)
PYEOF
# 4) Cosmos 3 Edge smoke test (20 frames, boxes drawn)
wait $DL
$PY -m det_eval.run vlm --model cosmos3_edge --smoke
$PY -m det_eval.run figures
echo PIPELINE_DONE
