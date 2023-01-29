from pycocotools.coco import COCO
from tqdm import tqdm
import os
import sys
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,BASE)
from utils.general import  Path
import argparse
import shutil
def json2labels(root,json_file,save_dir):
    # Make Directories
    dir = Path(root)  # dataset root dir
    for p in  'labels':
        (dir / p).mkdir(parents=True, exist_ok=True)
        for q in 'train', 'val':
            (dir / p / q).mkdir(parents=True, exist_ok=True)

    # print(os.listdir(os.path.join(root,'images',save_dir)))
    #print('./images/'+save_dir+'/'+im["file_name"])
    with open(os.path.join(root,save_dir+'.txt'),'w') as f:
        for file_name in os.listdir(os.path.join(root,'images',save_dir)):
            f.write('./images/'+save_dir+'/'+file_name+'\n')
    # Labels
    # for split in json_dict:
    coco = COCO(dir / 'annotations'/ json_file)
    names = [x["name"] for x in coco.loadCats(coco.getCatIds())]
    if not os.path.exists(dir / 'labels' / save_dir):
        os.makedirs(dir / 'labels' / save_dir)
    else:
        shutil.rmtree(dir / 'labels' / save_dir)
        os.makedirs(dir / 'labels' / save_dir)
    for cid, cat in enumerate(names):
        catIds = coco.getCatIds(catNms=[cat])
        imgIds = coco.getImgIds(catIds=catIds)
        for im in tqdm(coco.loadImgs(imgIds), desc=f'Class {cid + 1}/{len(names)} {cat}'):
            width, height = im["width"], im["height"]
            path = Path(im["file_name"])  # image filename
            try:
                with open(dir / 'labels' / save_dir / path.with_suffix('.txt').name, 'a') as file:
                    annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
                    for a in coco.loadAnns(annIds):
                        x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
                        x, y = x + w / 2, y + h / 2  # xy to center
                        file.write(f"{cid} {x / width:.5f} {y / height:.5f} {w / width:.5f} {h / height:.5f}\n")

            except Exception as e:
                print(e)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('--root', type=str,required=True) #../dataset/coco/
    parser.add_argument('--file', type=str,required=True) # train2017.json
    parser.add_argument('--save-dir', type=str,required=True) #train2017
    args = parser.parse_args()
    json2labels(
        args.root,
        args.file,
        args.save_dir
    )
