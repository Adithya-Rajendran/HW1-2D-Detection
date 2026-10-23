"""Small shared helpers: config/JSON IO, manifests, COCO id tables."""

from __future__ import annotations

import datetime as _dt
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Ultralytics/COCO 80-class index -> official COCO category id (91-id scheme).
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, indent: int | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
    return path


def read_ids(path: str | Path) -> list[str]:
    """Read a split file (one KITTI frame id per line)."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def git_state() -> dict:
    """Commit hash and dirty flag of this repo, if git is available."""
    def _run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": _run("rev-parse", "HEAD"), "dirty": bool(_run("status", "--porcelain"))}
    except Exception:
        return {"commit": None, "dirty": None}


def write_manifest(out_path: str | Path, **fields: Any) -> Path:
    """Write `<out_path>.manifest.json` next to an output: settings, command, git state, time."""
    out_path = Path(out_path)
    manifest = {
        "output": str(out_path),
        "created": now_iso(),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "git": git_state(),
        **fields,
    }
    return save_json(manifest, out_path.with_name(out_path.name + ".manifest.json"), indent=2)


def image_size(path: str | Path) -> tuple[int, int]:
    """(width, height) without decoding the whole image."""
    from PIL import Image

    with Image.open(path) as im:
        return im.size
