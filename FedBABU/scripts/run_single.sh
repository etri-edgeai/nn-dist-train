#!/bin/bash

### CNN ###

# # SGD, momentum 0.9 (Full-Body-Head order)
python main_single.py --dataset cifar10 --model cnn --num_classes 10 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.1 --body_m 0.9 --head_m 0.9 --results_save single
python main_single.py --dataset cifar10 --model cnn --num_classes 10 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save single
python main_single.py --dataset cifar10 --model cnn --num_classes 10 --epochs 160 --opt SGD --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.9 --results_save single

### Initialization
python main_single.py --dataset cifar10 --model cnn --num_classes 10 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save xavier_uniform
python main_single.py --dataset cifar10 --model cnn --num_classes 10 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save xavier_normal
python main_single.py --dataset cifar10 --model cnn --num_classes 10 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save kaiming_uniform
python main_single.py --dataset cifar10 --model cnn --num_classes 10 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save kaiming_normal
python main_single.py --dataset cifar10 --model cnn --num_classes 10 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save orthogonal
python main_single.py --dataset cifar10 --model cnn --num_classes 10 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save not_orthogonal
