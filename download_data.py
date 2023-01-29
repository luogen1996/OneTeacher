# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
import os


from utils.general import  check_dataset

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    check_dataset(opt.data)
    if 'VOC' in opt.data or 'voc' in opt.data:
        os.system('cp ./active_sampling/train.txt ../datasets/VOC')
        os.system('cp ./active_sampling/val.txt ../datasets/VOC')
        if not os.path.exists('../datasets/VOC2007'):
            os.system('ln -s ../datasets/VOC/images/VOCdevkit/VOC2007 ../datasets/VOC2007')
        if not os.path.exists('../datasets/VOC2012'):
            os.system('ln -s ../datasets/VOC/images/VOCdevkit/VOC2012 ../datasets/VOC2012')
