#!/usr/bin/env bash
# DGX Spark / ASUS Ascent GX10: Ultralytics' arm64 image (NVIDIA PyTorch, CUDA 13, TensorRT) with the
# repo mounted at /work, per Ultralytics' DGX Spark guide. Run from the repo root.
# The VLM arm (Cosmos 3 Nano) needs its own venv: torch 2.10 cu130 + transformers>=5.11 (see README).
set -euo pipefail
t=ultralytics/ultralytics:latest-nvidia-arm64
docker pull "$t"
docker run -it --rm --ipc=host --device nvidia.com/gpu=all -v "$PWD":/work -w /work "$t" \
  bash -lc 'pip install -q -e ".[dev]" && exec bash'
