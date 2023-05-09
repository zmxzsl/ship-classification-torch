#!/bin/bash

python main.py --train_mode 'Vanilla' --alpha '00' --model_name 'vgg11_bn' --gpus '0' & python main.py --train_mode 'Vanilla' --alpha '00' --model_name 'vgg13_bn' --gpus '1'  & python main.py --train_mode 'Vanilla' --alpha '00' --model_name 'vgg19_bn' --gpus '3'
python main.py --train_mode 'Vanilla' --alpha '00' --model_name 'vgg16_bn' --gpus '0'

