# One Flag and Two Gaps: What Zero-Shot COCO-to-Driving Detection Results Hide

CMPE 249 Intelligent Autonomous Systems (SJSU, Fall 2026), 2D object detection homework, **Track A (experimental)**. The assignment text is on the course Canvas page (not reproduced here); the report is in [`paper/`](paper/) (`latexmk -pdf main.tex`).

A devkit-faithful, tested harness for evaluating COCO-pretrained detectors zero-shot and fine-tuned on driving data (KITTI, nuImages), with paired bootstrap tests, a typed error census and Jetson Orin Nano latency.

## Findings

| | Finding | Evidence |
|---|---|---|
| 1 | **One flag decides the NMS vs. NMS-free comparison in AP.** Ultralytics' `predict()` runs NMS on the arg-max class; its validator uses `multi_label=True`. Turning the flag on adds 1.6 AP on KITTI, matches the NMS-free head's truck/tram recall, beats it by 0.9 AP (paired CI excludes zero), and closes most of the gap to published COCO numbers. At conf 0.25 it changes little. On nuImages it adds 0.6 AP / 5.1 AR (point estimate). | Paper Tables IV, V, X; `results/paired/`, `results/metrics/yolo26m_nms_ml__*` |
| 2 | **The COCO-to-KITTI gap is two gaps.** 1280 px input helps mainly cars under 40 px; 7 hours of fine-tuning on the Orin Nano helps cars above 25 px (car AP 50.7 to 68.3) but not smaller ones. On nuImages the resolution gain again shrinks with size but reaches mid-size objects. | Paper Fig. 2, Table X; height bins in `results/metrics/*__full.json` |
| 3 | **Many "false positives" are missing labels.** 60 to 70% of audited tiny false positives are real, unlabeled objects. | `results/audit_unlabeled_small.json` |
| | **The KITTI fine-tune does not transfer.** On nuImages it falls from 34.0 to 9.6 shared-class AP: 92% precise at conf 0.25 but 16% recall. Masking Van regions during training raises KITTI truck AP from 8.0 to 18.1 (one run per arm). | Paper Tables VI, X |
| | **LoRA on Cosmos 3 Edge learns the rare classes the fine-tuned detector loses.** Full KITTI val: 30.1 AP (JSON adapter) and 28.6 (compact) vs. 26.5 for fine-tuned YOLO26m; F1 0.83 / 0.82 / 0.82. Trucks 47.7 vs. 5.6 KITTI moderate AP. Orin Nano: 6.3 s/frame (compact) vs. 18 ms. Model size and pretraining are confounded, so this is not a LoRA-vs-full-fine-tune result. | Paper Table VII; `results/metrics/cosmos3_edge*`, `results/vlm_latency_jetson/` |

Also measured: closed-set vs. open-vocabulary (YOLOE-26m) vs. VLM (Cosmos 3 Edge, 17 s/frame zero-shot on the Orin Nano), TensorRT latency and memory, per-class AP, a failure gallery, and the course tutorial's own harness under three protocol switches.

## Assignment requirements map

| Requirement (Track A) | Where |
|---|---|
| 3+ configurations, 2+ families | Paper Table I; `configs/experiments.yaml` |
| 2+ datasets, taxonomy mapping | COCO, KITTI, nuImages; `configs/kitti_to_coco.yaml`, `configs/nuimages_to_coco.yaml`; Tables II, X |
| Metrics (AP, AP50/75, S/M/L, per-class, AR, latency, memory) | Tables IV, VI, IX, X |
| Reproducibility (checkpoint, split, resolution, thresholds, hardware, versions, commands) | Appendix Table XII (runs, settings, machines, commands), `runs/*/*.manifest.json`, `results/vlm_gx10/` |
| 2+ technical analyses | All five: cross-dataset, architecture, closed vs. open-vocabulary, speed-accuracy, fine-tuning |
| Failure analysis | Section V, Table XI, Fig. 5 |
| Evidence the experiments ran | Run logs and manifests in `runs/`, metrics in `results/`, tests in `tests/` |

## Released weights

| Weights | Hugging Face | License |
|---|---|---|
| Cosmos 3 Edge KITTI LoRA adapters (`json/`, `compact/`) | [Adithya-Rajendran/cosmos3-edge-kitti-lora](https://huggingface.co/Adithya-Rajendran/cosmos3-edge-kitti-lora) | OpenMDW-1.1 (from Cosmos 3 Edge); KITTI-trained, non-commercial |
| YOLO26m fine-tuned on KITTI (plain and Van-masked) | [Adithya-Rajendran/yolo26m-kitti](https://huggingface.co/Adithya-Rajendran/yolo26m-kitti) | AGPL-3.0 (from Ultralytics YOLO26); KITTI-trained, non-commercial |

Each model card has the metrics, usage code and training details. The code in this repository is MIT; the weights are not.

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

Raw outputs of every run: [`results/ngdet_repro/`](results/ngdet_repro/).

Protocol variants run on a separate patched copy outside the tutorial repository (it reproduces the first row exactly with the switches off). 1,000 frames, `--score-thr 0.001`, mAP in %; ML is multi-label NMS, NF the NMS-free head:

| Protocol | YOLO26x NMS | YOLO26x ML | YOLO26x NF | RT-DETR-R50 | Faster R-CNN v2 |
|---|---|---|---|---|---|
| `driving3` (tutorial) | 24.2 | 24.3 | 23.9 | 24.4 | 23.4 |
| + DontCare as ignore | 24.9 | 24.9 | 24.7 | 25.0 | 24.5 |
| `driving5` | 23.8 | **24.4** | 23.8 | 23.0 | 22.8 |
| + DontCare as ignore | 24.3 | **25.0** | 24.4 | 23.5 | 23.7 |

DontCare-as-ignore (KITTI devkit semantics) adds 0.5 to 1.1 to every model; multi-label NMS matters only once car and truck are separate classes (+0.6 mAP, +7.7 AR); cyclist AP is 0.0 for every model under both taxonomies because a COCO bicycle box never matches a KITTI rider-plus-bicycle box.

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

- **Datasets:** COCO val2017 (5,000 images, in-domain), the KITTI 2D object val split (3,769 frames, [`configs/splits/`](configs/splits/)) and 3,000 random nuImages val key frames (seed 0, `configs/splits/nuimages_val3k.txt`). The VLM comparison scores every model on the same fixed 300-frame KITTI subset.
- **Hardware:** NVIDIA Jetson Orin Nano 8GB (JetPack 6) for every detector run, the YOLO fine-tunes and every latency number, including the VLM's. The ASUS Ascent GX10 (NVIDIA GB10, 128 GB) trained the Cosmos LoRA adapters and produced all VLM accuracy numbers (greedy decoding keeps the machines close but not identical: the zero-shot run differs by about 1% in true positives, with the same F1). Each manifest records the machine.
- Everything is driven by [`configs/experiments.yaml`](configs/experiments.yaml).

## Evaluation protocol

**One evaluator for every model.** pycocotools scores COCO-format JSON for every model, with the same thresholds throughout: NMS IoU 0.7 and at most 300 boxes per image at prediction; conf 0.001 and the top 100 detections per image for AP; conf 0.25 for operating points and latency.

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
- 95% percentile bootstrap intervals over images: 200 resamples for single detectors, 300 for VLM runs, 300 to 500 for paired differences (`det_eval.compare`). With unit weights the bootstrap reproduces pycocotools exactly (tested).

**VLMs** return no confidence scores. For AP and AR each box is scored by the geometric-mean probability of the tokens that spell it; because those scores are not calibrated like a detector's, VLMs are also compared by precision, recall and F1 at IoU 0.5 and 0.7 with every box. Detectors are compared at conf 0.25 and threshold-free (AP).

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

### Docker on the Jetson (used for all new runs)

```bash
docker build -f docker/Dockerfile.jetson -t hw2d:jp6 .        # detectors, fine-tuning
docker build -f docker/Dockerfile.cosmos -t hw2d-cosmos:jp6 . # + Transformers 60f4954, peft 0.21.2 for the VLM
scripts/docker_run.sh python -m det_eval.run predict          # repo mounted at /work, runs as your user
```

### ASUS Ascent GX10 / DGX Spark (VLM fine-tuning)

A venv with torch 2.10 (cu130), `pip install -e ".[vlm]"`, Transformers at commit 60f49549e0db78f022ee30aa1571d9ab6e4a15bf and peft 0.21.2. `bash scripts/spark_docker.sh` gives Ultralytics' arm64 image with the repo mounted.

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
data/kitti/annotations/kitti_val_coco.json, kitti_val_coco_openvocab.json   (written by `run gt`)
data/nuimages/v1.0-val/*.json, data/nuimages/samples/CAM_*/            (official nuImages download)
```

## Run order

```bash
# Zero-shot detectors (Orin Nano)
python -m det_eval.run env                                  # hardware/software report
python -m det_eval.run gt                                   # COCO-format KITTI GT (closed-set and open-vocab)
python -m det_eval.nuimages subset --n 3000 --seed 0 --out configs/splits/nuimages_val3k.txt
python -m det_eval.nuimages gt --split configs/splits/nuimages_val3k.txt --out data/nuimages_coco/nuimages_val3k_coco.json
python -m det_eval.nuimages gt --split configs/splits/nuimages_val3k.txt --open-vocab --out data/nuimages_coco/nuimages_val3k_coco_openvocab.json
python -m det_eval.run predict                              # every detector x dataset x input size in the config
python -m det_eval.run predict --dataset nuimages_val3k
python -m det_eval.run evaluate                             # metrics for every prediction file and VLM run
python -m det_eval.run latency --engine                     # TensorRT FP16 (drop --engine for PyTorch)

# Fine-tuning YOLO26m on KITTI (Orin Nano, Docker)
python -m det_eval.finetune split
python -m det_eval.finetune export                          # ablation: --mask Van Person_sitting --out data/kitti_yolo_maskvan
python -m det_eval.finetune train --hours 7                 # ablation: --data data/kitti_yolo_maskvan/kitti5.yaml --name yolo26m_kitti5_640_maskvan
python -m det_eval.run predict --only yolo26m_ft_kitti yolo26m_ft_kitti_e2e yolo26m_ft_kitti_maskvan

# Cosmos 3 Edge (GX10 for accuracy, Orin Nano in Docker for latency)
python -m det_eval.run vlm --model cosmos3_edge --smoke     # 20 frames, boxes drawn in runs/vlm_smoke/cosmos3_edge/
python -m det_eval.vlm_finetune train --epochs 4 --grad-accum 8 --val-every 200 --n-val 100 --out runs/vlm_ft/cosmos3_edge_kitti
python -m det_eval.vlm_finetune train --epochs 4 --grad-accum 8 --val-every 200 --n-val 100 --format compact --order size --out runs/vlm_ft/cosmos3_edge_kitti_compact
python -m det_eval.vlm_grounding --model cosmos3_edge_ft                           # 300-frame subset; also cosmos3_edge, cosmos3_edge_ftc
python -m det_eval.vlm_grounding --model cosmos3_edge_ft --split configs/splits/kitti_val.txt   # full val
python -m det_eval.vlm_grounding --model cosmos3_edge_ftc --max-new-tokens 128 --out runs/vlm_budget/cosmos3_edge_ftc_b128__kitti_val__300.json
bash scripts/jetson_vlm_latency.sh                          # Orin latency, 50 frames, all three VLM configs

# Analyses
python -m det_eval.compare --gt data/kitti/annotations/kitti_val_coco.json --a runs/preds/A.json --b runs/preds/B.json --out results/paired/A_vs_B.json
python -m det_eval.failures --gt ... --dets ... --img-ids configs/splits/kitti_val_vlm300.txt --out results/failures.json
python -m det_eval.run report && python -m det_eval.run figures && python -m det_eval.gallery
python -m det_eval.paper_tables
cd paper && latexmk -pdf -outdir=build main.tex

# Course tutorial (ngdet) reproduction and protocol variants, outside the tutorial repo
python scripts/make_ngdet_patched.py ~/DeepDataMiningLearning ~/ngdet_patched
bash scripts/ngdet_variants.sh                              # outputs in results/ngdet_repro/
```

Every stage skips outputs that already exist; add `--force` to redo one, `--only MODEL ...` to restrict it. After the smoke run, open the drawn frames and confirm the boxes sit on the objects before the full run.

## Reproducibility

Every output file gets a `<name>.manifest.json`. It records the exact command, all settings (checkpoint, input size, thresholds, precision, NMS mode), and, for detector runs, the software and device versions; it also records the git commit. Most experiments ran before this directory became a git repository, so their recorded commit is null (the later Orin VLM latency runs record one); the tag `v1.0` marks the code and results behind the submitted paper. `python -m det_eval.run env` writes the hardware/software report.

Raw predictions go under `runs/` (not committed; large). Committed as evidence: metric files (`results/metrics/`), paired tests (`results/paired/`), the error census and label audit (`results/failures.json`, `results/audit_unlabeled_small.json`), Orin VLM latency summaries (`results/vlm_latency_jetson/`), the course-harness outputs (`results/ngdet_repro/`), and from `runs/` (force-added past .gitignore): prediction manifests (`runs/preds/*.manifest.json`), every latency run with per-frame timings (`runs/latency/`), Ultralytics validator runs (`runs/ultra_val/`), fine-tune arguments and per-epoch metrics (`runs/finetune/*/args.yaml`, `results.csv`), Orin VLM run manifests and summaries (`runs/vlm/`), the GX10 VLM runs (raw model outputs, summaries and manifests for the subset, full-val and token-cap runs, LoRA training logs and adapter manifests, and the run queue log: `results/vlm_gx10/`), the environment report (`runs/env/`), and the experiment logs and driver scripts (`runs/logs/`, minus download logs).

`scripts/docker_run.sh` and `scripts/jetson_vlm_latency.sh` mount `/mnt/bench-data` (a local NFS share) only if it exists; override with `DATA_MNT` and `ADAPTER_SRC`.

## Layout

```
configs/      experiments.yaml (single source of truth), kitti_to_coco.yaml, nuimages_to_coco.yaml, splits/
src/det_eval/ kitti.py, nuimages.py (GT conversion), evaluate.py (all metrics), compare.py (paired bootstrap),
              failures.py (error census), predict_ultralytics.py, ultra_val.py, latency.py, finetune.py,
              vlm_grounding.py, vlm_finetune.py (LoRA), report.py, plots.py, gallery.py, paper_tables.py,
              envinfo.py, run.py (orchestrator)
docker/       Dockerfile.jetson, Dockerfile.cosmos
scripts/      docker_run.sh, jetson_vlm_latency.sh, make_ngdet_patched.py, ngdet_variants.sh,
              jetson_system_setup.sh (sudo), jetson_setup.sh, spark_docker.sh, get_coco.sh
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
