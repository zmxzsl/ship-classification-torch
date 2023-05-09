#!/bin/bash

python main.py --train_mode 'additional' --alpha '02' --model_name 'vgg11_bn' --gpus '0' & python main.py --train_mode 'additional' --alpha '02' --model_name 'vgg13_bn' --gpus '1' & python main.py --train_mode 'additional' --alpha '02' --model_name 'vgg16_bn' --gpus '2' & python main.py --train_mode 'additional' --alpha '02' --model_name 'vgg19_bn' --gpus '3'

