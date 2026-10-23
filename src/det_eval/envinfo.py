"""Hardware and software report for the reproducibility table (run once per environment).

Usage:
    python -m det_eval.envinfo                 # prints JSON and writes runs/env/env_<timestamp>.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path

from .common import git_state, now_iso, save_json

PACKAGES = ("torch", "torchvision", "ultralytics", "transformers", "vllm", "pycocotools", "numpy",
            "opencv-python", "tensorrt", "accelerate")


def _cmd(*args: str) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except Exception:
        return None


def collect() -> dict:
    info: dict = {
        "created": now_iso(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "git": git_state(),
        "packages": {},
    }
    for pkg in PACKAGES:
        try:
            info["packages"][pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception:
        pass
    smi = _cmd("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")
    if smi:
        info["nvidia_smi"] = smi
    for release in ("/etc/dgx-release", "/etc/os-release"):
        p = Path(release)
        if p.exists():
            info[p.name] = dict(
                line.split("=", 1) for line in p.read_text().splitlines() if "=" in line
            )
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        total_kb = next((int(l.split()[1]) for l in meminfo.read_text().splitlines() if l.startswith("MemTotal")), None)
        if total_kb:
            info["mem_total_gib"] = round(total_kb / 2**20, 1)
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m det_eval.envinfo", description=__doc__)
    ap.add_argument("--out", default=f"runs/env/env_{dt.datetime.now():%Y%m%d_%H%M%S}.json")
    args = ap.parse_args(argv)
    info = collect()
    save_json(info, args.out, indent=2)
    print(json.dumps(info, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
