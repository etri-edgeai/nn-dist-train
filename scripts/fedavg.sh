#! /bin/bash

python train.py --dataset 'dirichlet_cifar10' --model 'resnet8' --clients-per-round 4 --num-rounds 1 --non-iid 100 --num-epochs 5