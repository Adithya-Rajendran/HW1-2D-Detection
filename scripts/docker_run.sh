#!/usr/bin/env bash
# Run a command in the Jetson container with the repo at /work, as the calling user (no root-owned files).
# KITTI zips are mounted read-only from NFS if present. Run from the repo root.
set -euo pipefail
IMAGE=${IMAGE:-hw2d:jp6}
DATA_MNT=${DATA_MNT:-/mnt/bench-data}
extra=()
[ -d "$DATA_MNT" ] && extra+=(-v "$DATA_MNT:$DATA_MNT:ro")
exec docker run --rm --runtime nvidia --ipc=host \
  -u "$(id -u):$(id -g)" -e HOME=/work/runs/.home -e USER="$(id -un)" \
  -e TORCHINDUCTOR_CACHE_DIR=/work/runs/.home/inductor \
  -v "$PWD:/work" "${extra[@]}" \
  ${DOCKER_NAME:+--name "$DOCKER_NAME"} "$IMAGE" "$@"
