import os
import json
import argparse
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


# Defining Connection Relationships
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

def InitCanvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas
    
def visualization(img_name, kps_list, save_pth):
    img = cv2.imread("./images/" + img_name)
    canvas = InitCanvas(img.shape[1], img.shape[0])
    #canvas = img
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
            
            #cv2.circle(img, pt1, 2, color_points[i//3], 2)
            cv2.circle(canvas, (tmp_list[0], tmp_list[1]), 3, color_points[i//3], -1)
            label_points.append(i/3)
            coor_lines.append(pt1)

        for noo, connect in enumerate(SHIP_PARTS):
            if ((connect[0]-1) in label_points) and ((connect[1]-1) in label_points):      
                p1_index = label_points.index(connect[0]-1)
                p2_index = label_points.index(connect[1]-1)
                
                pt1 = coor_lines[p1_index]
                pt2 = coor_lines[p2_index]
            
                cv2.line(canvas, pt1, pt2, color_lines[noo], 2)
                #cv2.line(img, pt1, pt2, color_lines[noo], 2)
    im_AB = np.concatenate([cv2.resize(img,(256,256)), cv2.resize(canvas,(256,256))], 1)
    cv2.imwrite(save_pth +img_name, im_AB)

    
if __name__ == "__main__":
    color_points = [[242, 12, 12], [242, 70, 12], [242, 127, 12], [242, 184, 12], [242, 242, 12], [184, 242, 12], [127, 242, 12], [70, 242, 12], [12, 242, 12], [12, 242, 70], [12, 242, 127], [12, 242, 184], [12, 242, 242], [12, 184, 242], [12, 127, 242], [12, 70, 242], [12, 12, 242], [70, 12, 242], [127, 12, 242], [184, 12, 242], [242, 12, 242], [242, 12, 184], [242, 12, 127], [242, 12, 70]]
    color_lines = [[253, 22, 22], [245, 59, 16], [254, 102, 11], [244, 152, 32], [252, 192, 13], [249, 234, 10], [223, 247, 53], [180, 251, 26], [146, 252, 40], [85, 249, 11], [45, 254, 15], [40, 249, 53], [48, 249, 99], [28, 243, 122], [23, 248, 164], [44, 253, 214], [34, 251, 251], [25, 203, 243], [44, 170, 245], [21, 122, 252], [36, 89, 249], [51, 63, 245], [75, 51, 246], [102, 36, 249], [148, 44, 252], [186, 52, 247], [222, 42, 248], [251, 43, 238], [250, 15, 191], [247, 33, 153], [252, 50, 126], [253, 27, 69], [242, 12, 127], [242, 12, 70]]

    save_pth = '../pytorch-CycleGAN-and-pix2pix/datasets/ships/'
    datadir = ["./anno/ships_keypoints_train2023.json", "./anno/ships_keypoints_val2023.json", "./anno/ships_keypoints_test2023.json"]
    mkdir(save_pth)
    for d_dir in datadir:
        data_mode = d_dir.split('_')[-1][:-9]
        mkdir(save_pth + data_mode)
        cocoData = load(d_dir)
        point_list = {}
        for ann2 in cocoData["annotations"]:
            ann = ann2.copy()
            img_id = ann["image_id"]
            imgName = '{:0>12d}.jpg'.format(img_id)
            if ann.__contains__("keypoints") == False:
                continue
            box_list = ann["bbox"]
            kp_list = ann["keypoints"]
            if imgName not in point_list.keys():
                point_list[imgName] = []
            point_list[imgName].append(kp_list)
        
        for img_name in point_list.keys():
            kps_list = point_list[img_name]
            visualization(img_name, kps_list, save_pth+data_mode+'/')
        
