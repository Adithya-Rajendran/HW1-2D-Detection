#!/usr/bin/env bash
# Jetson latency for Cosmos 3 Edge zero-shot and both KITTI LoRA adapters, inside Docker (host untouched).
# Same 50 frames (first 50 of the VLM subset) for all three, so the numbers compare directly.
# Adapters come from the Spark run (copied into runs/vlm_ft); the model comes from the host Hugging Face cache.
set -x
cd "$(dirname "$0")/.."
docker build -f docker/Dockerfile.cosmos -t hw2d-cosmos:jp6 . || exit 1
SRC=${ADAPTER_SRC:-/mnt/bench-data/datasets/cmpe249-hw1/spark/vlm_ft}
for a in cosmos3_edge_kitti cosmos3_edge_kitti_compact; do
  mkdir -p runs/vlm_ft/$a && cp -rn "$SRC/$a/adapter_best" runs/vlm_ft/$a/
done
mkdir -p runs/vlm_jetson
run() {
  docker run --rm --runtime nvidia --ipc=host -u "$(id -u):$(id -g)" \
    -e HOME=/work/runs/.home -e USER="$(id -un)" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
    -v "$PWD:/work" -v "$HOME/.cache/huggingface:/hf" hw2d-cosmos:jp6 "$@"
}
for m in cosmos3_edge_ftc cosmos3_edge_ft cosmos3_edge; do
  run python -m det_eval.vlm_grounding --model $m --limit 50 --out runs/vlm_jetson/${m}__kitti_val__50.json
done
echo JETSON_VLM_LATENCY_DONE
