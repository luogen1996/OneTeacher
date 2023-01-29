import os
import xml.etree.ElementTree as ET

# classes = ["bicycle","bus","motorbike","car","person"]
# classes =["person","bird","cat","cow","dog","horse","sheep","bottle","chair","dining table","potted plant","couch","tv","airplane","bicycle","boat","bus","car","motorcycle","train"]
# classes =["bicycle","bus","motorcycle","car","person"]
def xyxy2xywh(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2 * dw
    y = (box[1] + box[3]) / 2 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh
    return (x, y, w, h)

def voc2yolo(path,classes,outpath):
    print(len(os.listdir(path)))
    for file in os.listdir(path):
        label_file = path + file
        out_file = open(outpath + '/'+file.replace('xml', 'txt'), 'w')
        tree = ET.parse(label_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            bndbox = obj.find('bndbox')
            box = [float(bndbox.find('xmin').text), float(bndbox.find('ymin').text), float(bndbox.find('xmax').text),
                float(bndbox.find('ymax').text)]
            bbox = xyxy2xywh((w, h), box)
            out_file.write(str(cls_id) + " " + " ".join(str(x) for x in bbox) + '\n')

def getCOCOTxt(path):
    list_data1 = os.listdir(path + '/images/train2017/')
    list_data2 = os.listdir(path + '/images/val2017/')
    file = open(path+'/train2017.txt', 'w+')
    for i in list_data1:
        file.write('./images/train2017/'+i+'\n')
    file.close()
    file2 = open(path + '/val2017.txt', 'w+')
    for i in list_data2:
        file2.write('./images/val2017/' + i + '\n')
    file2.close()


def getVOCTxt(path,imagepath1, imagepath2, imagepath3):
    list_data1 = os.listdir(os.readlink(imagepath1))
    list_data2 = os.listdir(os.readlink(imagepath2))
    list_data3 = os.listdir(os.readlink(imagepath3))
    file = open(path + '/trainval2007.txt', 'w+')
    for i in list_data1:
        file.write('./images/trainval2007/' + i + '\n')
    file.close()
    file = open(path + '/trainval2012.txt', 'w+')
    for i in list_data2:
        file.write('./images/trainval2012/' + i + '\n')
    file.close()
    file = open(path + '/test2007.txt', 'w+')
    for i in list_data3:
        file.write('./images/test2007/' + i + '\n')
    file.close()

def run(dataset_name):
    basepath = '../datasets/'+dataset_name
    if 'coco' in dataset_name:
        dirs1 = basepath + '/labels/train2017'
        dirs2 = basepath + '/labels/val2017'
        if not os.path.exists(dirs1):
            os.makedirs(dirs1)
        if not os.path.exists(dirs2):
            os.makedirs(dirs2)
        path1= basepath +'/annotations/train2017/'
        path2= basepath +'/annotations/val2017/'
        if dataset_name =='coco5':
            classes = ["bicycle", "bus", "motorcycle", "car", "person"]
        elif dataset_name =='coco20':
            classes =["person","bird","cat","cow","dog","horse","sheep","bottle","chair","dining table","potted plant","couch","tv","airplane","bicycle","boat","bus","car","motorcycle","train"]
        voc2yolo(path1, classes,dirs1)
        voc2yolo(path2, classes,dirs2)
        getCOCOTxt(basepath)
    elif 'VOC_5' in dataset_name:
        dirs1 = basepath + '/labels/trainval2007'
        dirs2 = basepath + '/labels/trainval2012'
        dirs3 = basepath + '/labels/test2007'
        if not os.path.exists(dirs1):
            os.makedirs(dirs1)
        if not os.path.exists(dirs2):
            os.makedirs(dirs2)
        if not os.path.exists(dirs3):
            os.makedirs(dirs3)
        path1 = '../datasets/VOC2007_5/Annotations/trainval/'
        path2 = '../datasets/VOC2012_5/Annotations/trainval/'
        path3 = '../datasets/VOC2007_5/Annotations/test/'
        classes = ["bicycle", "bus", "motorbike", "car", "person"]
        voc2yolo(path1, classes,dirs1)
        voc2yolo(path2, classes,dirs2)
        voc2yolo(path3, classes,dirs3)
        imagepath1 = basepath + '/images/trainval2007'
        imagepath2 = basepath + '/images/trainval2012'
        imagepath3 = basepath + '/images/test2007'
        getVOCTxt(basepath,imagepath1,imagepath2,imagepath3)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert label2yolo')
    parser.add_argument("--dataset_name", type=str, default='coco20')
    args = parser.parse_args()
    run(args.dataset_name)
