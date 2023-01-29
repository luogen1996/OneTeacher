#!/bin/sh
if [ ! -d "./datasets" ]; then
    ln -s ../datasets ./datasets
fi
echo "start"
batch=$1
n_gpu=$2
devices=$3
dataseed=0,1,2,3,4,5,6,7,8,9
echo "check data"
python download_data.py --data ./data/coco.yaml

echo "generating random data for coco"
python active_sampling/generate_random_supervised_seed_yolo.py  --dataset_name 'coco_2017_train'  --random_seeds ${dataseed}  --random_file  ./data_processing/COCO_supervision.txt   --random_percent 10.0  --output_file ./dataseed/COCO_supervision_10.json

echo "training semi-yolov5s with 10\% labeled data"
python -m torch.distributed.launch --nproc_per_node $n_gpu  train_semi.py --batch $batch --data coco_semi.yaml --weights ''  --device ${devices}  --cfg ./models/yolov5s_semi_coco.yaml  --workers 12 --hyp data/hyps/hyp.semi.yaml --name yolo5s_semi_10_data

echo "Evaluation"
python val.py  --weights ./runs/train/yolo5s_semi_40_data/weights/best.pt --data coco_semi.yaml
echo "end"
