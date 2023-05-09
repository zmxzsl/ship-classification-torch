import os 
import random 
import time
import json
import torch
import torchvision
import numpy as np 
import pandas as pd 
import warnings
import argparse
from datetime import datetime
from torch import nn,optim
from configs.config import DefaultConfigs 
from collections import OrderedDict
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split,StratifiedKFold
from timeit import default_timer as timer
from models.model import *
from utils.utils import *
from utils.progress_bar import *
from IPython import embed
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

config = DefaultConfigs()

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

     
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name',
                        help='model directory',
                        type=str,
                        default='resnet50')
    parser.add_argument('--train_mode',
                        help='The way to train networks patterns',
                        type=str,
                        default='00')
    parser.add_argument('--alpha',
                        help='Adjust the highlight level of the keypoint area',
                        type=str,
                        default='00')
    parser.add_argument('--datasets',
                        help='selection train datasets or val datasets',
                        type=str,
                        default='val')
    parser.add_argument('--gpus',
                        help='Select GPU id',
                        type=str,
                        default='0`	')
    parser.add_argument('--start_idx',
                        help='selection visulization model',
                        type=int,
                        default=0)
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--images_path',
                        type=str,
                        default='./data/images/',
                        help='Input image path')
    parser.add_argument('--save_path',
                        type=str,
                        default='./cam/',
                        help='Input image path')
    parser.add_argument('--model_path',
                        type=str,
                        default='./checkpoints_region/',
                        help='Trianed Model Weight path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth',
                        action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args
def open_json(json_pth):
    with open(json_pth, 'r') as f:
        content = json.load(f)   
    return content
  
if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print('use GPUS ', args.gpus)
    config.model_name = args.model_name
    models_list = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152' , 'vgg16', 'vgg16_bn', \
                   'vgg19', 'vgg19_bn', 'squeezenet1_0' , 'densenet161', 'mobilenet_v2' , \
                   'mobilenet_v3_large', 'mobilenet_v3_small' , 'shufflenet_v2_x1_0', 'mnasnet1_0']
    
    methods = \
        {"fullgrad": FullGrad}
    '''
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,

         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    '''
    data_list = []

    # load dataset
    mk = open_json('./data/anno/category.json')
    mk_imos = list(mk.keys())[:30]


    test_datasets = open_json('./data/anno/class_{}.json'.format('val'))
    imgs_lists = os.listdir('./data/images')  
    for i in test_datasets['images']:
        imo = i['ship_info']['img_IMO'][0]
        if imo in mk_imos:
            #print(i)
            if i["file_name"] in imgs_lists:
                data_list.append([i['file_name'],mk[imo]])
    print('datasets images loaded!!!!!!!!!!!!!!!!!!!!')
    

    model_name = args.model_name
    model, target_layers = get_net_target(config, model_name)  
    
    for i in range(100,101):
    
        if i%5 != 0 and i>5:
            continue
  

        model_weight_pth = args.model_path + model_name + '/02/{}_checkpoint.pth.tar'.format(i)
        if i == 100:
           model_weight_pth = args.model_path + model_name + '/02/_checkpoint.pth.tar'    
        if not os.path.exists(model_weight_pth):
            print(model_name,' weights file not present!!!')
            continue

        model.load_state_dict(torch.load(model_weight_pth)["state_dict"])  
        model.cuda() 
        print('################',iter,model_name,' model loaded success')


        
        for iter, img_info in enumerate(tqdm(data_list)):
            imo_idx = img_info[1]-1
            img_name = img_info[0]
            img_idx = img_name[:-4]
            img_path = args.images_path + img_name


            rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
            rgb_img = np.float32(cv2.resize(rgb_img, (config.img_weight,config.img_height))) / 255
             
            input_tensor = preprocess_image(rgb_img,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                            

            # We have to specify the target we want to generate
            # the Class Activation Maps for.
            # If targets is None, the highest scoring category (for every member in the batch) will be used.
            # You can target specific categories by
            # targets = [e.g ClassifierOutputTarget(281)]
            
            input_tensor = Variable(input_tensor).cuda()
            
            output = model(input_tensor)
            smax = nn.Softmax(1)
            smax_out = smax(output)
            save_dir = args.save_path+img_idx + '/' + 'epoch{}'.format(i)+ '/'
            label = np.argmax(smax_out.cpu().data.numpy())
            mkdir(save_dir)

            
            targets = None
            for method in methods.keys():
                cam_algorithm = methods[method]
                if os.path.exists(f'{save_dir}{method}/{img_idx}_{model_name}_epoch{i}_cam_grayscale.npy'):
                    continue
                mkdir(f'{save_dir}{method}')
                #import ipdb;ipdb.set_trace()    
                if os.path.exists(f'{save_dir}{method}/{img_idx}_{model_name}_epoch{i}_cam_grayscale.jpg'):
                    continue
                with cam_algorithm(model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda) as cam:
                    cam.batch_size = 128
                    grayscale_cam = cam(input_tensor=input_tensor,
                                        targets=targets,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)

                    grayscale_cam = grayscale_cam[0, :]

                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
                gb = gb_model(input_tensor, target_category=None)

                cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
                cam_gb = deprocess_image(cam_mask * gb)
                gb = deprocess_image(gb)
                cv2.imwrite(f'{save_dir}{method}/{img_idx}_{model_name}_epoch{i}_{method}_cam.jpg', cam_image)
                cv2.imwrite(f'{save_dir}{method}/{img_idx}_{model_name}_epoch{i}_{method}_gb.jpg', gb)
                cv2.imwrite(f'{save_dir}{method}/{img_idx}_{model_name}_epoch{i}_{method}_cam_gb.jpg', cam_gb)
                cv2.imwrite(f'{save_dir}{method}/{img_idx}_{model_name}_epoch{i}_gt{imo_idx}_predict{label}_cam_grayscale.jpg', cam_mask*255)
                np.save(f'{save_dir}{method}/{img_idx}_{model_name}_epoch{i}_cam_grayscale.npy', grayscale_cam)
                print("image's gt:", imo_idx, "image's predict:",label)


