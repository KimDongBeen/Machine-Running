# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
from os import getcwd
import os
import cv2

# 파일 경로 입력
classes = ['geurae','sangsik','baekki','youngyi','dongsik']
train_path = "C:\\misaeng_data\\trainval_xml_filelist.txt"
imgsets_path_trainval = "C:\\misaeng_data\\trainval_img\\"
imgsets_path_test = "C:\\misaeng_data\\test_img\\"
test_path = "C:\\misaeng_data\\test_xml_filelist.txt"

def convert_annotation(TYPE):

    if TYPE=='trainval':
        imgsets_path = imgsets_path_trainval
        xml_path = train_path
    else:
        imgsets_path = imgsets_path_test
        xml_path = test_path

    f = open(xml_path, 'r')
    lines = f.readlines()

    print('Parsing annotation files')


    filename = "{}_misaeng.txt".format(TYPE)
    list_file = open(filename,"w")

    for idx, line in enumerate(lines):
        print('%d / %d'%(idx,len(lines)))

        et = ET.parse(line[:-1])
        root = et.getroot()

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            list_file.write(os.path.join(imgsets_path, line.split('\\')[-1][:-4] + "jpg"))
            img_file = os.path.join(imgsets_path, line.split('\\')[-1][:-4] + "jpg")
            test = cv2.imread(img_file)
            if type(test)=='Nonetype':
                print(img_file)
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
            list_file.write('\n')
    list_file.close()


def parse_misaeng_annotation(ann_dir, img_dir, labels=[]):
    # if os.path.exists(cache_name):
    #     with open(cache_name, 'rb') as handle:
    #         cache = pickle.load(handle)
    #     all_insts, seen_labels = cache['all_insts'], cache['seen_labels']

    all_insts = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        try:
            tree = ET.parse(ann_dir + ann)
        except Exception as e:
            print(e)
            print('Ignore this bad annotation: ' + ann_dir + ann)
            continue

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text + '.jpg'
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if os.path.isfile(img['filename']) is False:
            continue

        if len(img['object']) > 0:
            all_insts += [img]

    # cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
    # with open(cache_name, 'wb') as handle:
    #     pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_insts, seen_labels

# 데이터 집합 생성
convert_annotation('trainval')
convert_annotation('test')
