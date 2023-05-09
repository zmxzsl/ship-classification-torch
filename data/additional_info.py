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
mkdir('./add_info/')
for alpha in range(1,11):
    save_pth = './add_info/{}/'.format(str(alpha).zfill(2))
    mkdir(save_pth)
    beat = 0.1*alpha
    for img_name in os.listdir('./mask'):
        if not os.path.exists('./images/'+img_name):
            print(img_name, ' Missing RGB Image. ')
        if os.path.exists(save_pth+img_name):
            continue
        img = cv2.imread('./images/' + img_name)
        sk = cv2.imread('./mask/' + img_name)
        m = cv2.addWeighted(img,1,sk,beat,0)
        cv2.imwrite(save_pth+img_name,m)
