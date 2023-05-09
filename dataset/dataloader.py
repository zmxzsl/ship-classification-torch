from torch.utils.data import Dataset
from torchvision import transforms as T 
from PIL import Image
from itertools import chain 
from glob import glob
from tqdm import tqdm
from .augmentations import get_train_transform,get_test_transform
import random 
import numpy as np 
import pandas as pd 
import os 
import cv2
import torch 
import json


def open_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        ann = json.load(f)
    return ann

class ShipKeyPointsDataset(Dataset):
    def __init__(self, label_list, train=True, test=False):
        self.test = test 
        self.train = train 
        imgs = []
        if self.test:
            for index, row in label_list.iterrows():
                imgs.append((row["filename"], row["label"], row["keypoints"]))
            self.imgs = imgs 
        else:
            for index, row in label_list.iterrows():
                imgs.append((row["filename"], row["label"], row["keypoints"]))
            self.imgs = imgs

    def __getitem__(self, index):
        if self.test:
            filename, label, keypoints = self.imgs[index]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img,(int(config.img_height*1.5),int(config.img_weight*1.5)))
            img = cv2.resize(img, (int(config.img_height), int(config.img_weight)))
            img = get_test_transform(img.shape)(image=img)["image"]

            skeleton = cv2.imread(keypoints)
            skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img,(int(config.img_height*1.5),int(config.img_weight*1.5)))
            skeleton = cv2.resize(skeleton, (int(config.img_height), int(config.img_weight)))
            skeleton = get_test_transform(skeleton.shape)(image=skeleton)["image"]
            return img, label, skeleton, filename
        else:
            filename, label, keypoints = self.imgs[index]
           
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img,(int(config.img_height*1.5),int(config.img_weight*1.5)))
            img = cv2.resize(img, (int(config.img_height), int(config.img_weight)))
            img = get_train_transform(img.shape, augmentation=config.augmen_level)(image=img)["image"]

            skeleton = cv2.imread(keypoints)
            skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img,(int(config.img_height*1.5),int(config.img_weight*1.5)))
            skeleton = cv2.resize(skeleton, (int(config.img_height), int(config.img_weight)))
            skeleton = get_train_transform(skeleton.shape, augmentation=config.augmen_level)(image=skeleton)["image"]

            return img, label, skeleton, filename

    def __len__(self):
        return len(self.imgs)


class ShipDataset(Dataset):
    def __init__(self, label_list, config, args, test=False):
        self.test = test 
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.config = config
        imgs = []
        if self.test:
            for index, row in label_list.iterrows():
                imgs.append((row["filename"], row["label"], row["keypoints"]))
            self.imgs = imgs 
        else:
            for index, row in label_list.iterrows():
                imgs.append((row["filename"], row["label"], row["keypoints"]))
            self.imgs = imgs

    def __getitem__(self, index):
        if self.test:
            filename, label, keypoints = self.imgs[index]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img,(int(config.img_height*1.5),int(config.img_weight*1.5)))
            img = cv2.resize(img, (int(self.config.img_height), int(self.config.img_weight)))
            img = get_test_transform(img.shape, self.config)(image=img)["image"]
            
            return img, label, keypoints, filename
        else:
            filename, label, keypoints = self.imgs[index]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img,(int(config.img_height*1.5),int(config.img_weight*1.5)))
            img = cv2.resize(img, (int(self.config.img_height), int(self.config.img_weight)))
            img = get_train_transform(img.shape, self.config, augmentation=self.config.augmen_level)(image=img)["image"]
            return img, label, keypoints, filename

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    imgs = []
    label = []
    keypoints = []
    files_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        keypoints.append(sample[2])
        files_name.append(sample[3])

    return torch.stack(imgs, 0), label, keypoints, files_name


def collate_fn1(batch):
    imgs = []
    label = []
    keypoints = []
    files_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), label


def collate_fn2(batch):
    imgs = []
    label = []
    keypoints = []
    files_name = []
    for sample in batch:
        imgs.append(torch.cat((sample[0],sample[2]),0))
        label.append(sample[1])
    return torch.stack(imgs, 0), label


def collate_fn2rgb(batch):
    imgs = []
    label = []
    keypoints = []
    files_name = []
    for sample in batch:
        imgs.append(torch.cat((sample[0],sample[0]),0))
        label.append(sample[1])
    return torch.stack(imgs, 0), label


def get_keypoints(anns, img_info):
    keypoints = []
    for i in img_info:
        img_id = int(i.split('/')[-1][:-4])
        for q in anns:
            if img_id == q['image_id']:
                keypoints.append(q['keypoints'])
    return keypoints


def get_category_id(cate_list, cate_name):
    return cate_list[cate_name]-1


def get_files(root, mode):
    assert mode in ("train", "val")
    all_data_path, labels, keypoints = [], [], []
    files = open_json('./data/anno/class_{}.json'.format(mode))
    cate_list = open_json('./data/anno/category.json')

    for ann in files['images']:
        all_data_path.append(root+ann['file_name'])
        labels.append(get_category_id(cate_list, ann['ship_info']['img_IMO'][0]))
    keypoints = get_keypoints(files['annotations'], all_data_path)
    all_files = pd.DataFrame({"filename": all_data_path, "label": labels, "keypoints": keypoints})
    
    return all_files
