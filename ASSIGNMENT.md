# Homework: 2D Object Detection (Track A: Experimental Model Evaluation)

**Course:** CMPE 249 Intelligent Autonomous Systems, SJSU, Fall 2026 (Prof. Kaikai Liu)
**Due:** Monday, November 2, 2026, 11:59 pm
**Track:** A (Experimental). This repo follows Track A only.

> Copied from the Canvas assignment. Track B (code and architecture analysis) is omitted because this repo follows Track A. Em-dashes in the original are replaced with colons and the layout is adapted to Markdown; the wording is otherwise unchanged.

## Objective

The goal of this homework is to develop a technical understanding of modern 2D object detection. Ref our tutorial as the starting point: [ngdet TUTORIAL.md](https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md)

Choose ONE track based on your available computing resources:

- **Track A: Experimental Track.** If you have access to a GPU, run and evaluate modern object detection models. Training/fine-tuning is encouraged but not required; meaningful inference and cross-dataset evaluation are acceptable.

AI tools are allowed and encouraged, but your submission must demonstrate real technical investigation and understanding.

## Track A: Experimental Model Evaluation

Choose this track if you have access to a GPU, cloud GPU, or other suitable accelerator.

### Requirements

#### 1. Models

Evaluate at least 3 model configurations from at least 2 different detector families.

Examples:

- Faster R-CNN / Mask R-CNN
- RetinaNet / FCOS
- YOLOv8 / YOLO11
- DETR / Deformable DETR / RT-DETR
- Grounding DINO / OWLv2 / YOLO-World

Three sizes of the same model do not count as three detector families.

#### 2. Datasets

Evaluate on at least 2 datasets or domains.

Recommended datasets include:

- COCO
- KITTI
- nuImages
- nuScenes
- Waymo

A recommended experiment is:

> COCO-pretrained detector → autonomous-driving dataset without retraining

This allows you to study domain shift and generalization.

If datasets use different class definitions, clearly document your label/taxonomy mapping.

#### 3. Metrics

Report appropriate metrics such as:

- mAP@[0.50:0.95]
- AP50 / AP75
- AP small / medium / large
- per-class AP
- AR
- inference latency / FPS
- GPU memory, if measurable

#### 4. Reproducibility

For each major experiment, report:

- model and checkpoint
- dataset and split
- input resolution
- confidence/NMS thresholds
- hardware
- framework/version
- exact command or configuration

Screenshots alone are not sufficient.

#### 5. Technical Analysis

Perform at least two of the following:

- Cross-dataset generalization: Which classes or object scales degrade most?
- Architecture comparison: e.g., YOLO vs. RT-DETR or Faster R-CNN vs. FCOS.
- Closed-set vs. open-vocabulary: e.g., YOLO vs. Grounding DINO/YOLO-World.
- Speed-accuracy tradeoff: Compare latency, memory, and accuracy.
- Fine-tuning: Compare pretrained/zero-shot performance with fine-tuned performance.

## References

For either track, use at least 8 credible technical references, including:

- at least 4 original research papers
- at least 2 official repositories/framework documentation sources

Foundational papers are acceptable when discussing architectural evolution, but include recent models where appropriate.

## AI Tool Usage

ChatGPT, Claude, Codex, and other AI tools are allowed and encouraged.

Good uses of AI include:

- understanding unfamiliar source code
- locating model components
- comparing a paper with its implementation
- debugging experiments
- generating scripts to inspect tensor shapes
- analyzing unexpected results
- challenging your technical explanation

You are responsible for verifying AI-generated explanations against the actual paper, code, and experimental results.

## Submission

Submit one PDF as either:

- a 6-10 page technical report, or
- a 15-25 slide technical presentation.

Track A should include:

- Models and datasets
- Experimental setup and hardware
- Commands/configurations
- Quantitative results
- At least two technical analyses
- Failure analysis
- Conclusions
- References

Also submit/link your experiment code, configuration, or command log.

## Minimum Quality Requirement

Track A: You must provide evidence that you actually ran the experiments. Copied benchmark numbers or detection screenshots alone are not sufficient.

I am not grading based on who has the most powerful GPU.

For Track A, the key question is:

> Can you design, reproduce, and critically analyze a meaningful object-detection experiment?
