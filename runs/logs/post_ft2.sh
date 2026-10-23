set -x
cd ~/Projects/SJSU/Fall26/CMPE249/HW-2D-Detection
until grep -q POST_FT_DONE runs/logs/post_ft.log; do sleep 60; done
PYTHONUNBUFFERED=1 .venv/bin/python -m det_eval.run predict --only yolo26m_nms_ml
PYTHONUNBUFFERED=1 .venv/bin/python -m det_eval.run evaluate --only yolo26m_nms_ml
echo POST_FT2_DONE
