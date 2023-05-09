#!/bin/bash

python main.py --train_mode 'Vanilla' --alpha '00' --model_name 'resnet18' --gpus '0' & python main.py --train_mode 'Vanilla' --alpha '00' --model_name 'resnet34' --gpus '0'  & python main.py --train_mode 'Vanilla' --alpha '00' --model_name 'resnet50' --gpus '1' & python main.py --train_mode 'Vanilla' --alpha '00' --model_name 'resnet101' --gpus '2' & python main.py --train_mode 'Vanilla' --alpha '00' --model_name 'resnet152' --gpus '3'

