#!/usr/bin/env bash
# One-time system setup on the Jetson Orin Nano (needs sudo). Run interactively:
#     sudo bash scripts/jetson_system_setup.sh
# Then run scripts/jetson_setup.sh as your normal user for the Python environment.
#
# Sources: Ultralytics NVIDIA Jetson guide (cudss for the torch 2.10 wheel, max power, jetson_clocks)
# and Jetson AI Lab "RAM Optimization" (disable zram, swap file on NVMe).
set -euo pipefail
SWAP_GB="${SWAP_GB:-16}"

# Max power mode: the MAXN id differs per board (Orin Nano Super: 0=15W, 1=25W, 2=MAXN_SUPER)
MAXN_ID=$(grep -oP 'POWER_MODEL ID=\K[0-9]+(?= NAME=MAXN)' /etc/nvpmodel.conf | head -1)
nvpmodel -m "$MAXN_ID"          # persists across reboots; may ask to reboot
nvpmodel -q
jetson_clocks                   # does NOT persist: re-run after every reboot, before timing anything

apt-get update
apt-get install -y python3-pip python3-venv cudss rsync

# zram compresses RAM; it adds no room for model weights. Use a real swap file on the NVMe instead.
systemctl disable nvzramconfig || true   # takes effect after a reboot
if ! swapon --show | grep -q '^/swapfile'; then
  fallocate -l "${SWAP_GB}G" /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
swapon --show
echo "Optional, frees about 0.8 GB for the VLM run: sudo systemctl set-default multi-user.target && sudo reboot"
