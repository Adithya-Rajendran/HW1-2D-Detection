set -x
cd ~/HW1-2D-Detection
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -m det_eval.vlm_finetune train --max-steps 30 --grad-accum 8 --val-every 30 --n-val 20 --out runs/vlm_ft/smoke_json
.venv/bin/python -m det_eval.vlm_finetune train --max-steps 30 --grad-accum 8 --val-every 30 --n-val 20 --format compact --order size --out runs/vlm_ft/smoke_compact
echo SMOKE_DONE
