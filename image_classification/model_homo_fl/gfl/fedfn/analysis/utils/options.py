#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    
    
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--hetero_option', type=str, default='iid', help="client heterogenity option")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--alpha', type=float, default=0.1, help="lda parameter")
    
    
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=100, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--wd', type=float, default=0.0, help="weight decay (default: 0.0)")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--scheduler', type=str, default='multistep', help="learning rate scheduling option")
    
    
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')

    # other arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')#local client에서의 accuracy, loss 성능 print할건지 여부
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    
    # additional arguments
    parser.add_argument('--local_upt_part', type=str, default=None, help='body, head, or full')
    
    # arguments for a single model
    parser.add_argument('--body_lr', type=float, default=None, help="learning rate for the body of the model")
    parser.add_argument('--head_lr', type=float, default=None, help="learning rate for the head of the model")

    #Algorithm name
    parser.add_argument('--fl_alg', type=str, default='FedAvg', help="federated learning algorithm")
    
    #Whether Feature normalizing
    parser.add_argument('--fn', action='store_true', help='whether feature normalized or not')
    
    #FedProx parameter
    parser.add_argument('--mu', type=float, default=1.0, help="parameter for proximal local SGD")

    #FedNTD, FedLSD parameter
    parser.add_argument('--tau', type=float, default= 3.0, help="softened_parameter")
    parser.add_argument('--w_kd', type=float, default=1.0, help="kd_weight")
    parser.add_argument('--w_ce', type=float, default=1.0, help="ce_weight")
    
    #Scaffold parameter
    parser.add_argument('--adaptive_division', action='store_true', help="scaffold parameter")
    
    
    #feature norm parameter
    parser.add_argument('--feature_norm', type=int, default=1, help="featuire norm parameter")
    
    args = parser.parse_args()
    return args
