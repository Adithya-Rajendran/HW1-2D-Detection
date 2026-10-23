"""Make a patched copy of the ngdet harness for protocol variants. The original repo is never modified.

Switches (environment variables, all off by default so the copy behaves exactly like the original):
  NGDET_DONTCARE_IGNORE=1  KITTI DontCare boxes become COCO crowd (ignore) regions for every class,
                           as the KITTI devkit does, instead of being dropped (detections on them count as FPs).
  NGDET_YOLO_NMS=0|1       pass nms=<bool> to Ultralytics predict (selects the YOLO26 head explicitly).
  NGDET_MULTILABEL=1       run Ultralytics NMS with multi_label=True, as its validator does.
  NGDET_COUNT_NMS=1        print how many times NMS ran (to confirm which head predict() used).
"""
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(sys.argv[1]).expanduser()   # ~/Projects/ref/DeepDataMiningLearning
DST = Path(sys.argv[2]).expanduser()   # ~/Projects/ref/ngdet_patched
if DST.exists():
    shutil.rmtree(DST)
subprocess.run(["rsync", "-a", "--exclude", ".git", f"{SRC}/", f"{DST}/"], check=True)
PKG = DST / "DeepDataMiningLearning"


def patch(rel, old, new):
    p = PKG / rel
    s = p.read_text()
    n = s.count(old)
    assert n == 1, f"{rel}: expected 1 match, found {n}: {old[:60]!r}"
    p.write_text(s.replace(old, new))
    print("patched", rel)


# 1. KITTI loader: keep DontCare boxes in target["dontcare"] when the switch is on.
patch("detection/dataset_kitti.py",
      "        boxes = []\n        labels = []\n        for i in range(num_objs):\n",
      "        boxes = []\n        labels = []\n        dontcare = []  # ngdet_patched\n        for i in range(num_objs):\n")
patch("detection/dataset_kitti.py",
      "            if objecttype != 'DontCare' and xmax-xmin>0 and ymax-ymin>0:\n",
      "            if objecttype == 'DontCare' and xmax-xmin>0 and ymax-ymin>0 and "
      "__import__('os').environ.get('NGDET_DONTCARE_IGNORE') == '1':  # ngdet_patched\n"
      "                dontcare.append([xmin, ymin, xmax, ymax])\n"
      "            if objecttype != 'DontCare' and xmax-xmin>0 and ymax-ymin>0:\n")
patch("detection/dataset_kitti.py",
      "        num_objs = len(labels) #update num_objs\n        newtarget = {}\n",
      "        num_objs = len(labels) #update num_objs\n        newtarget = {}\n"
      "        newtarget['dontcare'] = torch.as_tensor(dontcare, dtype=torch.float32).reshape(-1, 4)  # ngdet_patched\n")

# 2. EvalSample carries the ignore boxes.
patch("ngdet/datasets.py",
      "    gt_labels: np.ndarray  # [M] unified ids\n",
      "    gt_labels: np.ndarray  # [M] unified ids\n    gt_ignore: object = None  # ngdet_patched: [K,4] xyxy ignore regions\n")
patch("ngdet/datasets.py",
      "        # Fold native GT labels into unified ids; drop classes outside taxonomy.\n",
      "        ign = target.get('dontcare') if self._mode == 'torch_target' else None  # ngdet_patched\n"
      "        if ign is not None:\n"
      "            ign = (ign.cpu().numpy() if hasattr(ign, 'cpu') else np.asarray(ign)).reshape(-1, 4)\n"
      "        # Fold native GT labels into unified ids; drop classes outside taxonomy.\n")
patch("ngdet/datasets.py",
      "            gt_labels=np.asarray(keep_l, np.int64),\n        )\n",
      "            gt_labels=np.asarray(keep_l, np.int64),\n            gt_ignore=ign,\n        )\n")

# 3. Evaluator: ignore regions become crowd annotations for every class.
patch("ngdet/evaluator.py",
      "        # Detections (COCO results format).\n",
      "        ign = getattr(sample, 'gt_ignore', None)  # ngdet_patched\n"
      "        for box in ([] if ign is None else ign):\n"
      "            xywh = self._xyxy_to_xywh(box)\n"
      "            for u in range(len(self.taxonomy.classes)):\n"
      "                self._gt_anns.append({'id': self._ann_id, 'image_id': img_id, 'category_id': u + 1,\n"
      "                                      'bbox': xywh, 'area': float(xywh[2] * xywh[3]), 'iscrowd': 1})\n"
      "                self._ann_id += 1\n\n"
      "        # Detections (COCO results format).\n")

# 4. YOLO adapter: explicit head, multi-label NMS, NMS call counter.
patch("ngdet/detectors/yolo.py",
      "        results = self.model.predict(\n"
      "            image, device=self.device, conf=0.0, verbose=False, half=self._half)[0]\n",
      "        import os  # ngdet_patched\n"
      "        from ultralytics.utils import nms as _nms\n"
      "        if not hasattr(_nms.non_max_suppression, '_orig'):\n"
      "            _orig = _nms.non_max_suppression\n"
      "            def _patched(*a, **k):\n"
      "                if os.environ.get('NGDET_MULTILABEL') == '1':\n"
      "                    k['multi_label'] = True\n"
      "                _patched.calls += 1\n"
      "                return _orig(*a, **k)\n"
      "            _patched._orig, _patched.calls = _orig, 0\n"
      "            _nms.non_max_suppression = _patched\n"
      "        extra = {}\n"
      "        if os.environ.get('NGDET_YOLO_NMS') in ('0', '1'):\n"
      "            extra['nms'] = os.environ['NGDET_YOLO_NMS'] == '1'\n"
      "        results = self.model.predict(\n"
      "            image, device=self.device, conf=0.0, verbose=False, half=self._half, **extra)[0]\n"
      "        if os.environ.get('NGDET_COUNT_NMS') == '1':\n"
      "            print(f'[ngdet_patched] nms calls so far: {_nms.non_max_suppression.calls}', flush=True)\n")
print("done:", DST)
