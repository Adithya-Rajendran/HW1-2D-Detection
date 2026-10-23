set -x
cd ~/Projects/ref
R="docker run --rm --runtime nvidia --ipc=host -u $(id -u):$(id -g) -e HOME=/home/jetson/Projects/ref/hf -e USER=jetson -e HF_HOME=/home/jetson/Projects/ref/hf -e YOLO_CONFIG_DIR=/home/jetson/Projects/ref/hf/yolo -e TORCHINDUCTOR_CACHE_DIR=/home/jetson/Projects/ref/hf/inductor -v /home/jetson/Projects:/home/jetson/Projects -v $PWD/DeepDataMiningLearning:/ref -w /home/jetson/Projects/ref/out ngdet:jp6 python -m DeepDataMiningLearning.ngdet.run_eval"
M="yolo:yolo26x.pt hf_detr:PekingU/rtdetr_r50vd torchvision:fasterrcnn_resnet50_fpn_v2"
K="--roots kitti=/home/jetson/Projects/ref/kitti_root/ --datasets kitti --device cuda"
# A: his exact setting (120 images, score-thr 0.3 default, driving3)
$R --models $M $K --max-images 120 --out-dir /home/jetson/Projects/ref/out/A_exact
# B: one change: keep low-score boxes for AP
$R --models $M $K --max-images 120 --score-thr 0.001 --out-dir /home/jetson/Projects/ref/out/B_thr001
# C: B on 1,000 images (sample size)
$R --models $M $K --max-images 1000 --score-thr 0.001 --out-dir /home/jetson/Projects/ref/out/C_thr001_n1000
echo NGDET_ABC_DONE
