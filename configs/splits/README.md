# KITTI splits

| File | Frames | Source |
|---|---|---|
| `kitti_train.txt` | 3,712 | Standard train half of KITTI's 7,481 labeled frames (Chen et al., NeurIPS 2015), copied from OpenPCDet `data/kitti/ImageSets/train.txt` on 2026-10-20 |
| `kitti_val.txt` | 3,769 | Matching val half, from OpenPCDet `data/kitti/ImageSets/val.txt` |
| `kitti_val_vlm300.txt` | 300 | `python -m det_eval.kitti subset --split configs/splits/kitti_val.txt --n 300 --seed 0` |

The two halves are disjoint and together cover all 7,481 frames. They are split by recording sequence, so near-duplicate consecutive frames do not leak between them. All models are evaluated zero-shot on `kitti_val`. `kitti_train` is only for the optional fine-tuning comparison.

Every model, VLMs included, is scored on the same 300-frame subset for the open-vocabulary comparison.

SHA-256:

```
e85ce0142be11c7e4196fd7b79a8bc8c2cefdd6fe754ac61fef8d421e37aba5c  kitti_train.txt
6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8  kitti_val.txt
379b54883bbf9558cd3517a3968752737e88944e4a8d421996fa65cc16ca8e86  kitti_val_vlm300.txt
```
