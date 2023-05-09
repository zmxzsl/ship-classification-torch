#!/bin/bash

git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
cd data
python raw_visualization.py
cp -r ./multi_view_ships/ ../pytorch-CycleGAN-and-pix2pix/datasets/multi_view_ships/
cd ../pytorch-CycleGAN-and-pix2pix/
python train.py --dataroot ./datasets/ships --name ships_pix2pix --model pix2pix --direction BtoA

# python test.py --dataroot ./datasets/ships --name ships_pix2pix --model pix2pix --direction BtoA --num_test 144 --epoch 190
# python test.py --dataroot ./datasets/multi_view_ships --name ships_pix2pix --model pix2pix --direction BtoA --num_test 144 --epoch 190