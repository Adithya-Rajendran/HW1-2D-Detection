set -x
cd ~/HW1-2D-Detection
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
# 1. two LoRA runs, identical budget: 4 epochs over 3,412 frames (grad-accum 8 -> 1,706 optimizer steps)
$P -m det_eval.vlm_finetune train --epochs 4 --grad-accum 8 --val-every 200 --n-val 100 --out runs/vlm_ft/cosmos3_edge_kitti
$P -m det_eval.vlm_finetune train --epochs 4 --grad-accum 8 --val-every 200 --n-val 100 --format compact --order size --out runs/vlm_ft/cosmos3_edge_kitti_compact
echo TRAIN_DONE
# 2. scored evaluation on the 300-frame subset (token-probability confidences -> AP/AR)
for m in cosmos3_edge_ftc cosmos3_edge_ft cosmos3_edge; do
  $P -m det_eval.vlm_grounding --model $m --out runs/vlm/${m}__kitti_val__300.json
done
# 3. speed: compile test and token-budget sweep on the compact adapter
$P -m det_eval.vlm_grounding --model cosmos3_edge_ftc --limit 20 --compile --out runs/vlm/cosmos3_edge_ftc_compile__kitti_val__20.json
for b in 64 128 256; do
  $P -m det_eval.vlm_grounding --model cosmos3_edge_ftc --max-new-tokens $b --out runs/vlm_budget/cosmos3_edge_ftc_b${b}__kitti_val__300.json
done
echo EVAL300_DONE
# 4. full validation split for the fine-tuned models
$P -m det_eval.vlm_grounding --model cosmos3_edge_ftc --split configs/splits/kitti_val.txt --out runs/vlm/cosmos3_edge_ftc__kitti_val__3769.json
$P -m det_eval.vlm_grounding --model cosmos3_edge_ft --split configs/splits/kitti_val.txt --out runs/vlm/cosmos3_edge_ft__kitti_val__3769.json
echo FULLVAL_DONE
mkdir -p /mnt/bench-data/datasets/cmpe249-hw1/spark && cp -r runs/vlm runs/vlm_budget runs/vlm_ft runs/logs /mnt/bench-data/datasets/cmpe249-hw1/spark/
echo COPY_DONE
