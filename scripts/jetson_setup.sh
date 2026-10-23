#!/usr/bin/env bash
# Python environment on the Jetson Orin Nano (JetPack 6.x, Python 3.10). No sudo needed.
# Run scripts/jetson_system_setup.sh (with sudo) once before this.
#
# Detector stack follows Ultralytics' NVIDIA Jetson guide: prebuilt torch 2.10.0 / torchvision 0.25.0
# cp310 wheels. Those are also the versions Cosmos 3's Transformers path pins, so the same venv runs
# the Cosmos 3 Edge reasoner. Run from the repo root:   bash scripts/jetson_setup.sh
set -euo pipefail
WHEELS=https://github.com/ultralytics/assets/releases/download/v0.0.0

[ -f /etc/nv_tegra_release ] || { echo "not a Jetson (no /etc/nv_tegra_release)"; exit 1; }
head -1 /etc/nv_tegra_release              # JetPack 6 is L4T R36.x
python3 -c 'import sys; assert sys.version_info[:2] == (3, 10), "the prebuilt wheels are cp310 (JetPack 6)"'
sudo -n true 2>/dev/null && echo "(sudo available)" || true
nvpmodel -q 2>/dev/null | head -1 || true  # should say MAXN_SUPER before any timing run

# --system-site-packages keeps JetPack's TensorRT Python bindings visible (needed for engine export)
[ -d .venv ] || python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -U pip
pip install -q "$WHEELS/torch-2.10.0-cp310-cp310-linux_aarch64.whl" \
               "$WHEELS/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl"
pip install -q -e ".[detect,dev]"
# Cosmos 3 Edge is only in Transformers main for now (per NVIDIA's Cosmos README)
pip install -q "transformers @ git+https://github.com/huggingface/transformers.git" accelerate "safetensors>=0.8.0"

python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", float((x @ x).float().abs().mean()) > 0)
try:
    import tensorrt
    print("tensorrt", tensorrt.__version__)
except ImportError:
    print("tensorrt python bindings not visible (engine export will fail)")
EOF
python -m pytest -q
python -m det_eval.envinfo --out runs/env/jetson_env.json > /dev/null && echo "wrote runs/env/jetson_env.json"
echo "done: source .venv/bin/activate"
