#! /bin/bash

model="vgg11_bn"
dataset="dirichlet_cifar10"
algorithm="fedavg_pdp"
non_iid="0.001"
participation="2"

for niid in $non_iid
do
    for part in $participation
    do
        CUDA_VISIBLE_DEVICES='6' python train.py --dataset $dataset --model $model --algorithm $algorithm --non-iid $niid --num-rounds 100 --num-clients 20 --clients-per-round $part --num-epochs 1 --niid-split 5 --exp-name 'num_split_5'
    done
done