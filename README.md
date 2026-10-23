# One Flag and Two Gaps: What Zero-Shot COCO-to-Driving Detection Results Hide

CMPE 249 Intelligent Autonomous Systems (SJSU, Fall 2026), 2D object detection homework, **Track A (experimental)**. The assignment is in [`ASSIGNMENT.md`](ASSIGNMENT.md), the report in [`paper/`](paper/) (`latexmk -pdf main.tex`).

A devkit-faithful, tested harness for evaluating COCO-pretrained detectors zero-shot and fine-tuned on driving data (KITTI, nuImages), with paired bootstrap tests, a typed error census and Jetson Orin Nano latency.

## Findings

| | Finding | Evidence |
|---|---|---|
| F1 | **One flag decides the NMS vs. NMS-free comparison.** Ultralytics' `predict()` runs NMS on the arg-max class; its validator uses `multi_label=True`. Turning the flag on adds 1.6 AP on KITTI, matches the NMS-free head's truck/tram recall, beats it by 0.9 AP, and closes most of the gap to published COCO numbers. | `results/paired/`, `results/metrics/yolo26m_nms_ml__*` |
| F2 | **The COCO-to-KITTI gap is two gaps.** 1280 px input helps only cars under 40 px; 7 hours of fine-tuning on the Orin Nano helps only cars above 25 px (car AP 50.7 to 68.3). | Paper Fig. 2, `results/metrics/*__full.json` height bins |
| F3 | **Many "false positives" are missing labels.** 60 to 70% of audited tiny false positives are real, unlabeled objects. | `results/audit_unlabeled_small.json` |

Also measured: closed-set vs. open-vocabulary (YOLOE-26m) vs. VLM (Cosmos 3 Edge, 17 s/frame on the Orin Nano), TensorRT latency and memory, per-class AP, and a failure gallery.

## Assignment requirements map

| Requirement (Track A) | Where |
|---|---|
| 3+ configurations, 2+ families | Paper Table I; `configs/experiments.yaml` |
| 2+ datasets, taxonomy mapping | COCO, KITTI, nuImages; `configs/kitti_to_coco.yaml`, `configs/nuimages_to_coco.yaml`; Table II |
| Metrics (AP, AP50/75, S/M/L, per-class, AR, latency, memory) | Tables III, V, VII |
| Reproducibility (checkpoint, split, resolution, thresholds, hardware, versions, commands) | Appendix tables IX and X, `runs/*/*.manifest.json` |
| 2+ technical analyses | All five: cross-dataset, architecture, closed vs. open-vocabulary, speed-accuracy, fine-tuning |
| Failure analysis | Section V, Table VIII, Fig. 5 |
| Evidence the experiments ran | Run logs and manifests in `runs/`, metrics in `results/`, tests in `tests/` |

## Citation

See [`CITATION.cff`](CITATION.cff). GitHub shows a "Cite this repository" button from it.

## Relation to the course tutorial (ngdet)

The starting point is the course's [ngdet tutorial](https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md) by Prof. Kaikai Liu. This repository does not fork or modify it; it is an independent harness, and the tutorial's KITTI run is reproduced with ngdet's own `run_eval` (unmodified, in Docker on the Jetson Orin Nano):

| KITTI, ngdet `driving3`, mAP | YOLO26x | RT-DETR-R50 | Faster R-CNN v2 |
|---|---|---|---|
| Tutorial (RTX 3090, 120 images) | 0.264 | 0.262 | 0.279 |
| Reproduced, same command | 0.236 | 0.240 | 0.236 |
| `--score-thr 0.001` | 0.248 | 0.248 | 0.238 |
| `--score-thr 0.001`, 1,000 images | 0.242 | 0.244 | 0.234 |

Variants that change the protocol (KITTI DontCare as ignore regions, a five-class taxonomy, multi-label NMS) run on a separate patched copy outside the tutorial repository; see the paper's reproduction section.

## Original questions

- **RQ1, scale vs appearance.** How much of a COCO detector's accuracy drop on KITTI comes from objects shrinking at the input resolution, and how much from the scenes looking different?
- **RQ2, suppression.** YOLO26 has two heads on the same weights. Does the NMS-free head lose more accuracy under domain shift, and what does it buy in latency?
- **RQ3, semantics.** COCO has no "cyclist". Can an open-vocabulary detector or a physical-AI VLM cover it, and at what localization quality and latency?

## Design

| Config | Family | Checkpoint |
|---|---|---|
| `yolo26m_nms` | Dense one-stage CNN, one-to-many head + NMS | `yolo26m.pt` |
| `yolo26m_nms_ml` | Same NMS head, `multi_label=True` (as Ultralytics' validator) | `yolo26m.pt` |
| `yolo26m_e2e` | Same weights, one-to-one head, NMS-free (`nms=False`) | `yolo26m.pt` |
| `yolo26m_ft_kitti` (+ `_e2e`, `_maskvan`) | YOLO26m fine-tuned on KITTI train (`det_eval.finetune`) | `runs/finetune/.../best.pt` |
| `rtdetr_l` | Query-based transformer, Hungarian matching, NMS-free | `rtdetr-l.pt` |
| `yoloe26m` | Open-vocabulary detector, text prompts | `yoloe-26m-seg.pt` loaded into `yoloe-26m.yaml` (detection head) |
| `cosmos3_edge` | Physical-AI VLM, 2D grounding (2.4B reasoner) | `nvidia/Cosmos3-Edge` |
| `cosmos3_nano` (optional) | Physical-AI VLM, 2D grounding (8B reasoner) | `nvidia/Cosmos3-Nano` |
| `qwen3vl_8b` (optional) | General VLM; Cosmos 3 Nano's base model | `Qwen/Qwen3-VL-8B-Instruct` |

- **Datasets:** COCO val2017 (5,000 images, in-domain) and the KITTI 2D object val split (3,769 frames, [`configs/splits/`](configs/splits/)). Every model is evaluated zero-shot. The VLM comparison scores every model on the same fixed 300-frame subset.
- **Hardware:** NVIDIA Jetson Orin Nano 8GB (JetPack 6) for all detector runs, latency and Cosmos 3 Edge. The ASUS Ascent GX10 (NVIDIA GB10, 128 GB) is only needed for the optional 8B VLMs.
- Everything is driven by [`configs/experiments.yaml`](configs/experiments.yaml).

## Evaluation protocol

**One evaluator for every model.** pycocotools scores COCO-format JSON for every model, with the same thresholds throughout: conf 0.001 and at most 100 detections per image for AP; conf 0.25 for operating points and latency.

**KITTI to COCO mapping** ([`configs/kitti_to_coco.yaml`](configs/kitti_to_coco.yaml)):

| KITTI class | Treatment |
|---|---|
| Car, Pedestrian, Truck, Tram | COCO car, person, truck, train |
| Van | Ignored for car, matched by IoU (the devkit's neighboring-class rule) |
| Person_sitting | Ignored for person, matched by IoU (same rule) |
| Cyclist | Ignored for person, matched by containment, because a COCO "person" box covers only the rider. It is a real class (`cyclist`) for open-vocabulary models and VLMs. |
| DontCare | Ignore region for every class, matched by intersection over detection area (devkit rule) |
| Misc | Dropped |

Closed-set models get a rule-based `cyclist` baseline built from overlapping person and bicycle detections.

**Metrics:**
- AP@[.5:.95], AP50, AP75, AP small/medium/large, AR, and per-class AP.
- AP by box height (<25, 25-40, 40-80, >=80 px).
- KITTI-style AP|R40 per difficulty at the devkit IoU (car 0.7, pedestrian and cyclist 0.5).
- 95% bootstrap intervals from 500 image resamples. With unit weights the bootstrap reproduces pycocotools exactly (tested).

**VLMs** return no confidence scores, so they get operating points instead of AP: precision, recall and F1 at IoU 0.5 and 0.7. Detectors are compared at their best F1 and at matched recall.

**Latency:**
- Batch 1, with images decoded before timing and the GPU synchronized around each call.
- Run at conf 0.25, reporting p50/p90/p99 end to end and per stage, against the number of GT objects per frame.
- Measured in both PyTorch and TensorRT FP16.
- Peak memory comes from `torch.cuda.max_memory_allocated()`, since `nvidia-smi` can't report it on unified-memory devices.

## Setup

### Jetson Orin Nano (JetPack 6)

```bash
sudo bash scripts/jetson_system_setup.sh   # once: MAXN power mode, cudss, 16 GB swap file on NVMe instead of zram
bash scripts/jetson_setup.sh               # venv with Ultralytics' Jetson torch wheels + Transformers main, runs the tests
source .venv/bin/activate
```

Before any timing run, `sudo nvpmodel -q` must report `MAXN_SUPER` (or `MAXN`). Mode IDs differ by board; see `/etc/nvpmodel.conf`. Then run `sudo jetson_clocks`. The power mode persists across reboots, but `jetson_clocks` does not.

With 8 GB of unified memory, run one job at a time: detectors first, then the VLM. For the VLM run, booting to console (`sudo systemctl set-default multi-user.target`) frees about 0.8 GB.

### ASUS Ascent GX10 / DGX Spark (optional)

Run `bash scripts/spark_docker.sh` for Ultralytics' arm64 image with the repo mounted.

Cosmos 3 Nano needs its own venv: torch 2.10 (cu130) and torchvision 0.25, then `pip install -e ".[vlm]"`.

## Data

```bash
bash scripts/get_coco.sh                         # COCO val2017 -> data/coco
# KITTI: register at cvlibs.net, download data_object_image_2.zip (12 GB) and data_object_label_2.zip,
# then extract only the val frames straight from the zips (about 3 GB):
python -m det_eval.kitti copy --split configs/splits/kitti_val.txt \
    --image-zip /path/to/data_object_image_2.zip --label-zip /path/to/data_object_label_2.zip
python -m det_eval.run gt                        # COCO-format GT, closed-set and open-vocab
```

Expected layout:

```
data/coco/images/val2017/*.jpg
data/coco/annotations/instances_val2017.json
data/kitti/training/image_2/*.png        (val frames only is enough)
data/kitti/training/label_2/*.txt
data/kitti/annotations/kitti_val_coco.json, kitti_val_coco_openvocab.json   (written by make gt)
```

## Run order

```bash
python -m det_eval.run env                                  # hardware/software report
python -m det_eval.run predict                              # every detector x dataset x input size in the config
python -m det_eval.run evaluate                             # metrics for every prediction file (and VLM runs)
python -m det_eval.run latency                              # PyTorch, batch 1, conf 0.25
python -m det_eval.run latency --engine                     # TensorRT FP16 engines
python -m det_eval.run vlm --model cosmos3_edge --smoke     # 20 frames, boxes drawn in runs/vlm_smoke/cosmos3_edge/
python -m det_eval.run vlm --model cosmos3_edge             # full 300-frame run
python -m det_eval.run evaluate && python -m det_eval.run report && python -m det_eval.run figures
cd paper && latexmk -pdf -outdir=build main.tex
```

Every stage skips outputs that already exist; add `--force` to redo one, `--only MODEL ...` to restrict it. After the smoke run, open the drawn frames and confirm the boxes sit on the objects before the full run.

## Reproducibility

Every output file gets a `<name>.manifest.json`. It records the exact command, git commit and dirty flag, all settings (checkpoint, input size, thresholds, precision, NMS mode) and the software and device versions. `make env` writes the hardware/software report used in the paper's reproducibility table.

Predictions and logs go under `runs/` (not committed). Metric summaries under `results/` are committed as evidence.

## Layout

```
configs/      experiments.yaml (single source of truth), kitti_to_coco.yaml, splits/
src/det_eval/ kitti.py (GT conversion, splits, frame copy), evaluate.py (all metrics),
              predict_ultralytics.py, latency.py, vlm_grounding.py, report.py, plots.py,
              envinfo.py, run.py (orchestrator)
scripts/      jetson_system_setup.sh (sudo), jetson_setup.sh, spark_docker.sh, get_coco.sh
tests/        synthetic KITTI frames covering every mapping and ignore rule
paper/        IEEE conference source, figures/, refs.bib
results/      committed metric summaries
```

## Key references

- KITTI: Geiger, Lenz, Urtasun, CVPR 2012. COCO: Lin et al., ECCV 2014. Val split: Chen et al., NeurIPS 2015.
- DETR (Carion et al., ECCV 2020), RT-DETR (Zhao et al., CVPR 2024), YOLOv10 (Wang et al., NeurIPS 2024), YOLOE (Wang et al., 2025).
- Cosmos 3: NVIDIA, arXiv:2606.02800 (2026).
- Ultralytics docs (YOLO26, RT-DETR, YOLOE, Jetson guide), cocoapi.

## License

Code: MIT ([`LICENSE`](LICENSE)). Datasets keep their own terms: COCO (CC BY 4.0 annotations), KITTI (CC BY-NC-SA 3.0), nuImages (CC BY-NC-SA 4.0). No dataset files are in this repository.
