#! /bin/bash

model="vgg11_bn"
dataset="dirichlet_cifar10"
algorithm="fedprox"
non_iid="0.001 0.1 10.0"
participation="6"
mu="0.001"

for niid in $non_iid
do
    for part in $participation
    do
        for m in $mu
        do
            CUDA_VISIBLE_DEVICES='5' python train.py --dataset $dataset --model $model --algorithm $algorithm --non-iid $niid --num-rounds 100 --num-clients 20 --clients-per-round $part --num-epochs 1 --mu $m
        done
    done
done