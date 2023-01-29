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
python download_data.py --data ./data/VOC.yaml

echo "generating random data for voc"
python data_processing/generate_random_supervised_seed_yolo.py  --dataset_name 'voc_2007_trainval+voc_2012_trainval'  --random_seeds ${dataseed}  --random_file  ./data_processing/voc07_voc12.txt   --random_percent 25.0  --output_file ./dataseed/VOC_supervision_25.json

echo "training semi-yolov5s with 25\% labeled data"
python -m torch.distributed.launch --nproc_per_node $n_gpu  train_semi.py --batch $batch --data VOC_semi.yaml --weights ''  --device ${devices}   --cfg ./models/yolov5s_semi_voc.yaml  --workers 12 --hyp data/hyps/hyp.semi_voc.yaml --name yolo5s_semi_25_data_voc --steps 50100 --val-per-epoch 10

echo "Evaluation"
python val.py  --weights ./runs/train/yolo5s_semi_25_data_voc/weights/best.pt --data VOC_semi.yaml
echo "end"