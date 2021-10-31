#!/bin/bash

# After FedAvg
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 20 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 8 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1

python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 20 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 8 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1

python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 80 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 20 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 8 --lr 0.001 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1

python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 20 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 8 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1

python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 20 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 8 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1

python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 80 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 20 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1
python main_lg.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 8 --lr 0.001 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --results_save run1 --momentum 0.90 --wd 0.0 --load_fed fed --num_layers_keep 1