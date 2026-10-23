#!/bin/bash
# Protocol variants of the ngdet KITTI run, on the patched copy (ngdet_patched); the original repo is untouched.
set -x
cd ~/Projects/ref
D="docker run --rm --runtime nvidia --ipc=host -u $(id -u):$(id -g) -e HOME=/home/jetson/Projects/ref/hf -e USER=jetson -e HF_HOME=/home/jetson/Projects/ref/hf -e YOLO_CONFIG_DIR=/home/jetson/Projects/ref/hf/yolo -e TORCHINDUCTOR_CACHE_DIR=/home/jetson/Projects/ref/hf/inductor -v /home/jetson/Projects:/home/jetson/Projects -v $PWD/ngdet_patched:/ref -w /home/jetson/Projects/ref/out"
P="ngdet:jp6 python -m DeepDataMiningLearning.ngdet.run_eval"
M3="yolo:yolo26x.pt hf_detr:PekingU/rtdetr_r50vd torchvision:fasterrcnn_resnet50_fpn_v2"
K="--roots kitti=/home/jetson/Projects/ref/kitti_root/ --datasets kitti --device cuda"
N="--max-images 1000 --score-thr 0.001"
o() { echo "--out-dir /home/jetson/Projects/ref/out/$1"; }
# 0: equivalence: patched copy, switches off, must equal A_exact
$D $P --models $M3 $K --max-images 120 $(o V0_patched_equals_A)
# driving3 (his taxonomy): YOLO26x head and NMS variants
$D -e NGDET_MULTILABEL=1 $P --models yolo:yolo26x.pt $K $N $(o C_ml)
$D -e NGDET_YOLO_NMS=0   $P --models yolo:yolo26x.pt $K $N $(o C_e2e)
# D: DontCare as ignore regions (KITTI devkit semantics), driving3
$D -e NGDET_DONTCARE_IGNORE=1 $P --models $M3 $K $N $(o D3)
$D -e NGDET_DONTCARE_IGNORE=1 -e NGDET_MULTILABEL=1 $P --models yolo:yolo26x.pt $K $N $(o D3_ml)
$D -e NGDET_DONTCARE_IGNORE=1 -e NGDET_YOLO_NMS=0   $P --models yolo:yolo26x.pt $K $N $(o D3_e2e)
# E: driving5 (car/truck/bus/person/cyclist), DontCare dropped as in the original
$D $P --taxonomy driving5 --models $M3 $K $N $(o E5)
$D -e NGDET_MULTILABEL=1 $P --taxonomy driving5 --models yolo:yolo26x.pt $K $N $(o E5_ml)
$D -e NGDET_YOLO_NMS=0   $P --taxonomy driving5 --models yolo:yolo26x.pt $K $N $(o E5_e2e)
# D+E: driving5 with DontCare ignored
$D -e NGDET_DONTCARE_IGNORE=1 $P --taxonomy driving5 --models $M3 $K $N $(o DE5)
$D -e NGDET_DONTCARE_IGNORE=1 -e NGDET_MULTILABEL=1 $P --taxonomy driving5 --models yolo:yolo26x.pt $K $N $(o DE5_ml)
$D -e NGDET_DONTCARE_IGNORE=1 -e NGDET_YOLO_NMS=0   $P --taxonomy driving5 --models yolo:yolo26x.pt $K $N $(o DE5_e2e)
echo NGDET_VARIANTS_DONE
