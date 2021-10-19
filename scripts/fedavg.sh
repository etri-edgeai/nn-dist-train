#! /bin/bash

model="vgg11_bn"
dataset="dirichlet_cifar10"
algorithm="fedavg"
non_iid="0.001 0.1 10.0"
participation="2"

for niid in $non_iid
do
    for part in $participation
    do
    CUDA_VISIBLE_DEVICES='0' python train.py --dataset $dataset --model $model --algorithm $algorithm --non-iid $niid --num-rounds 100 --num-clients 20 --clients-per-round $part --num-epochs 1
    done
done