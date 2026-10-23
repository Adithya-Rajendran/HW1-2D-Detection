set -x
cd ~/Projects/SJSU/Fall26/CMPE249/HW-2D-Detection
scripts/docker_run.sh python -m det_eval.run predict --only yolo26m_ft_kitti yolo26m_ft_kitti_e2e
scripts/docker_run.sh python -m det_eval.run evaluate --only yolo26m_ft_kitti yolo26m_ft_kitti_e2e
echo FT_EVAL_DONE
