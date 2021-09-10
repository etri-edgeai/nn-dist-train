#! /bin/bash

python train.py --dataset 'dirichlet_cifar10' --model 'resnet8' --num-clients 1 --clients-per-round 1 --num-rounds 100 --non-iid 0.01 --num-epochs 1