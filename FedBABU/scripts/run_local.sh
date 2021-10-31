#!/bin/bash

python main_local.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 160 --lr 0.1 --num_users 100 --results_save local
python main_local.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 160 --lr 0.1 --num_users 100 --results_save local
python main_local.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 160 --lr 0.1 --num_users 100 --results_save local