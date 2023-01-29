#!/bin/sh
batch=$1   #64
n_gpu=$2   #2
devices=$3 #0,1
python download_data.py --data ./data/VOC.yaml

python -m torch.distributed.launch --nproc_per_node $n_gpu  train.py --batch $batch --data VOC.yaml --weights ''  --device ${devices}   --cfg ./models/yolov5s.yaml  --workers 12 --hyp data/hyps/hyp.scratch_voc.yaml --name yolo5s_fully_sup_voc --step-lr --steps 50100 --val-per-epoch 10

echo "Evaluation"
python val.py  --weights ./runs/train/yolo5s_fully_sup_voc/weights/best.pt --data VOC.yaml