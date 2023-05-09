import os
import json
import cv2
from pycocotools.coco import COCO
import numpy as np


def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print (path+' Successful Creation!')
        return True
    else:
        print (path+' Catalogues already exist!')
        return False


SHIP_PARTS = [[1, 3], \
              [2, 3], [1, 2], [3, 4], \
              [3, 5], [3, 6], [4, 5], \
              [4, 6], [5, 6], [6, 7], \
              [6, 8], [7, 9], [8, 10], \
              [9, 11], [10, 12], [7, 11], \
              [8, 12], [11, 12], [11, 1], \
              [12, 2], [11, 13], [12, 14], \
              [13, 14], [13, 15], [14, 16], \
              [15, 16], [15, 17], [16, 18], \
              [17, 18], [17, 19], [18, 20], \
              [19, 20], [19, 1], [20, 2]]


def load(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def InitCanvas(width, height, color=(0, 0, 0)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas

def visualization(img_name, kps_list):

    name = 0

    img = cv2.imread("./images/" + img_name)
    canvas = InitCanvas(img.shape[1], img.shape[0])


    for kp_list in kps_list:
        tmp_list = []
        label_points = []
        coor_lines = []
        for i in range(0, len(kp_list), 3):    
            if (kp_list[i] == kp_list[i+1] == kp_list[i+2] == 0):
                continue
            tmp_list = []
            tmp_list.append(int(kp_list[i]))
            tmp_list.append(int(kp_list[i+1]))
            tmp_list.append(int(kp_list[i+2]))

            pt1 = (tmp_list[0], tmp_list[1])
            cv2.circle(canvas, (tmp_list[0], tmp_list[1]), 20, [255,255,255], -1)
            label_points.append(i/3)
            coor_lines.append(pt1)
        
    m = cv2.addWeighted(img,1,canvas,0.2,0)
    cv2.imwrite('./mask/' +img_name, canvas)



    
if __name__ == "__main__":
    datadir = ["./anno/class_train.json"]
    mkdir('./mask/')
    for d_dir in datadir:
        cocoData = load(d_dir)
        point_list = {}
        for ann2 in cocoData["annotations"]:
            ann = ann2.copy()
            img_id = ann["image_id"]
            imgName = '{:0>12d}.jpg'.format(img_id)
            if ann.__contains__("keypoints") == False:
                continue
            x, y, w, h = ann["bbox"]
            box_list = ann["bbox"]
            kp_list = ann["keypoints"]
            if imgName not in point_list.keys():
                point_list[imgName] = []
            point_list[imgName].append(kp_list)

        for img_name in point_list.keys():
            kps_list = point_list[img_name] 
            visualization(img_name, kps_list)
            
        
