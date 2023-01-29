from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
import json
from detectron2.data.build import (
    get_detection_dataset_dicts,
)
from builtin import *

def generate(dataset,random_file,random_percent):
    dataset = tuple([item for item in dataset.split(',')])
    dataset_dicts = get_detection_dataset_dicts(
        dataset,
        filter_empty=True,
    )

    try:
        with open(random_file,'r') as f:
            dic=json.load(f)
    except:
        dic={}

    dic[str(random_percent)] = {}
    seeds = [int(i) for i in args.random_seeds.split(',')]
    for i in range(10):
        arr = generate_supervised_seed(
            dataset_dicts,
            random_percent,
            seeds[i]
        )
        print(len(arr))
        dic[str(random_percent)][str(i)] = [dataset_dicts[idx]['file_name'].split('/')[-1] for idx in arr]
            
    with open(random_file,'w') as f:
        f.write(json.dumps(dic))

def generate_supervised_seed(
    dataset_dicts, SupPercent, seed
):
    num_all = len(dataset_dicts)
    num_label = int(SupPercent / 100.0 * num_all)

    arr = range(num_all)
    import random
    random.seed(seed)
    return random.sample(arr,num_label)

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--dataset_name",type=str,default='voc_2007_trainval_for_5cls,voc_2012_trainval_for_5cls')
    parser.add_argument("--random_seeds",type=str,default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--random_file",type=str,default='temp/voc07_5+voc12_5.json')
    parser.add_argument("--random_percent",type=float,default=15.0)
    args = parser.parse_args()
    generate(
        args.dataset_name,
        args.random_file,
        args.random_percent,
    )
