set -x
cd ~/Projects/SJSU/Fall26/CMPE249/HW-2D-Detection
while [ -n "$(docker ps -q)" ]; do sleep 10; done
Z=/mnt/bench-data/datasets/kitti
scripts/docker_run.sh python -m det_eval.kitti copy --split configs/splits/kitti_train.txt --image-zip $Z/data_object_image_2.zip --label-zip $Z/data_object_label_2.zip
ls data/kitti/training/image_2 | wc -l; ls data/kitti/training/label_2 | wc -l
scripts/docker_run.sh python -m pytest -q
scripts/docker_run.sh python -m det_eval.finetune split
scripts/docker_run.sh python -m det_eval.finetune export
echo FT_PREP_DONE
