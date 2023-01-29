#!/bin/sh
batch=$1   #64
n_gpu=$2   #2
devices=$3 #0,1
echo "check data"
python download_data.py --data ./data/coco.yaml

python -m torch.distributed.launch --nproc_per_node $n_gpu  train.py --batch $batch --data coco.yaml --weights ''  --device ${devices}  --cfg ./models/yolov5s.yaml  --workers 12 --hyp data/hyps/hyp.scratch_coco.yaml --name yolo5s_fully_sup_coco

echo "Evaluation"
python val.py  --weights ./runs/train/yolo5s_fully_sup_coco/weights/best.pt --data coco.yaml