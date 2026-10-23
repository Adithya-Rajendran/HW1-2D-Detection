#!/usr/bin/env bash
# COCO val2017 images and instances annotations into data/coco (about 1 GB). Run from the repo root.
set -euo pipefail

fetch() {  # fetch URL FILE, with whichever downloader is installed (a stock Jetson has wget, not curl)
  if command -v curl >/dev/null; then curl -L --fail -o "$2" "$1"; else wget -q --show-progress -O "$2" "$1"; fi
}

mkdir -p data/coco/images data/coco/annotations
cd data/coco
if [ ! -d images/val2017 ]; then
  fetch http://images.cocodataset.org/zips/val2017.zip val2017.zip
  unzip -q val2017.zip -d images && rm val2017.zip
fi
if [ ! -f annotations/instances_val2017.json ]; then
  fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip annotations.zip
  unzip -q -j annotations.zip annotations/instances_val2017.json -d annotations && rm annotations.zip
fi
echo "val2017 images: $(ls images/val2017 | wc -l) (expect 5000)"
