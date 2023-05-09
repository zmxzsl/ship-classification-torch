#!/bin/bash

cd data
python demo --pth './anno/class_train.json' & python demo --pth './anno/class_val.json'
python mask_visualization.py
python additional_info.py
cd ..
