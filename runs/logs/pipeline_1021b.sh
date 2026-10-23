set -x
cd ~/Projects/SJSU/Fall26/CMPE249/HW-2D-Detection
.venv/bin/python -m det_eval.run latency --engine
.venv/bin/python -m det_eval.run vlm --model cosmos3_edge --smoke
.venv/bin/python -m det_eval.run figures
echo PIPELINE2_DONE
