#! /bin/bash

model="vgg11_bn"
dataset="dirichlet_cifar10"
algorithm="fedavg_pdp"
non_iid="0.001"
participation="6" # 6 10

for niid in $non_iid
do
    for part in $participation
    do
        CUDA_VISIBLE_DEVICES='3' python train.py --dataset $dataset --model $model --algorithm $algorithm --non-iid $niid --num-rounds 100 --num-clients 20 --clients-per-round $part --num-epochs 1 --self-balancing --exp-name 'fedcsb'
    done
done