set -x
cd ~/Projects/SJSU/Fall26/CMPE249/HW-2D-Detection
scripts/docker_run.sh python -m det_eval.finetune export --out data/kitti_yolo_maskvan --mask Van Person_sitting
DOCKER_NAME=ftmask scripts/docker_run.sh python -m det_eval.finetune train --data data/kitti_yolo_maskvan/kitti5.yaml --name yolo26m_kitti5_640_maskvan --hours 7 --batch 8 --workers 3
scripts/docker_run.sh python -m det_eval.run predict --only yolo26m_ft_kitti_maskvan
scripts/docker_run.sh python -m det_eval.run evaluate --only yolo26m_ft_kitti_maskvan
echo MASKVAN_DONE
