#!/bin/bash

cd data
git clone https://github.com/jacobgil/pytorch-grad-cam/
mv ./pytorch-grad-cam/pytorch_grad_cam/ ../pytorch_grad_cam/
rm -rf ./pytorch-grad-cam/
cd ..