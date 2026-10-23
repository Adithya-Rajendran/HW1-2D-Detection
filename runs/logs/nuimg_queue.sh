set -x
cd ~/Projects/SJSU/Fall26/CMPE249/HW-2D-Detection
D=/mnt/bench-data/datasets/nuimages
# 1. wait for the samples download (wget exits), then verify and extract
while pgrep -x wget >/dev/null; do sleep 60; done
ls -la $D/nuimages-v1.0-all-samples.tgz
tail -c 400 $D/samples.log
gzip -t $D/nuimages-v1.0-all-samples.tgz && echo SAMPLES_OK
nice tar xzf $D/nuimages-v1.0-all-samples.tgz -C $D/root
n=0; for f in $(PYTHONPATH=src .venv/bin/python -c "import json;print(\"\\n\".join(i[\"file_name\"] for i in json.load(open(\"data/nuimages_coco/nuimages_val3k_coco.json\"))[\"images\"]))"); do [ -f data/nuimages/$f ] || n=$((n+1)); done; echo "missing subset images: $n"
# 2. wait for the masked-Van fine-tune queue to finish (GPU), then run nuImages
until grep -q MASKVAN_DONE runs/logs/maskvan.log; do sleep 120; done
PYTHONUNBUFFERED=1 .venv/bin/python -m det_eval.run predict --dataset nuimages_val3k --only yolo26m_nms yolo26m_nms_ml yolo26m_e2e rtdetr_l yoloe26m
scripts/docker_run.sh python -m det_eval.run predict --dataset nuimages_val3k --only yolo26m_ft_kitti yolo26m_ft_kitti_maskvan
PYTHONUNBUFFERED=1 .venv/bin/python -m det_eval.run evaluate --only yolo26m_nms yolo26m_nms_ml yolo26m_e2e rtdetr_l yoloe26m yolo26m_ft_kitti yolo26m_ft_kitti_maskvan
echo NUIMG_DONE
