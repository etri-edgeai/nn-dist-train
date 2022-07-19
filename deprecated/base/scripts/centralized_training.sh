#! /bin/bash

model="vgg11_bn"
dataset="dirichlet_cifar10"
algorithm="centralized"

python train.py --dataset $dataset --model $model --algorithm $algorithm --num-rounds 100