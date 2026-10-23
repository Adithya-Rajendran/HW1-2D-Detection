set -x
cd ~/Projects/SJSU/Fall26/CMPE249/HW-2D-Detection
while [ -n "$(docker ps -q --filter name=^ft$)" ]; do sleep 60; done
ls runs/finetune/yolo26m_kitti5_640/weights/
.venv/bin/python -m det_eval.ultra_val labels
.venv/bin/python -m det_eval.ultra_val val --name yolo26m_nms
.venv/bin/python -m det_eval.ultra_val val --name yolo26m_e2e --e2e
scripts/docker_run.sh python -m det_eval.run predict --only yolo26m_ft_kitti yolo26m_ft_kitti_e2e
scripts/docker_run.sh python -m det_eval.run evaluate --only yolo26m_ft_kitti yolo26m_ft_kitti_e2e
echo POST_FT_DONE
