import os, sys
import copy
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import itertools
from tqdm import tqdm

from utils.args import parse_args
from utils.device import gpu_to_cpu, cpu_to_gpu
from utils.util import fix_seed, make_log, modify_path
from data_utils import *
from models import *
from train_tools.client_opt import client_opt
from train_tools.server_opt import server_opt


DATASET = {}
MODEL = {}

def _get_args():
    # get argument for training
    args = parse_args()
    
    # make experiment reproducible
    fix_seed(args.seed)
        
    # model log file
    args.log_filename, args.log_pd, args.check_filename, args.img_filename = make_log(args)
    
    return args


def _make_dataloader(args):
    # create dataloader
    client_loader, dataset_sizes, args.class_num, args.num_clients = DATASET[args.dataset](args)
    
    # refine path in argument
    args.log_filename, args.check_filename, args.img_filename = modify_path(args)
    
    # size of each local client's local data
    client_datasize = np.zeros(args.num_clients)
    for client in range(args.num_clients): 
        client_datasize[client] += dataset_sizes['train'][client]    
    
    return client_loader, dataset_sizes, args
    
    
def _make_model(args):
    # create model for server and client
    model = MODEL[args.model](num_classes=args.num_classes, **args.model_kwargs) if args.model_kwargs else MODEL[args.model](num_classes=args.num_classes)
    
    # model to gpu
    model = model.to(args.device)
        
    # initialize server and client model weights
    server_weight = gpu_to_cpu(copy.deepcopy(model.state_dict()))
    for k, v in server_weight.items():
        server_weight[k] = v
    server_momentum = {}
    
    client_weight = {}
    client_momentum = {}
    for client in range(args.num_clients):
        client_weight[client] = gpu_to_cpu(copy.deepcopy(model.state_dict()))
        client_momentum[client] = {}
    
    weight = {'server': server_weight,
              'client': client_weight}
    momentum = {'server': server_momentum,
                'client': client_momentum}
    
    return model, weight, momentum

    
def train():
    args = _get_args()
    
    client_loader, dataset_sizes, args = _make_dataloader(args)
    
    model, model, weight, momentum = _make_model(args)

    # evaluation metrics
    selected_clients_num = np.zeros(args.num_clients)
    best_acc = [-np.inf, 0, 0, 0]
    test_acc = []
    
    # Federated Learning Pipeline
    for r in tqdm(range(args.num_rounds)):
        # client selection and updated the selected clients
        weight, momentum, selected_clients = client_opt(args, client_loader, client_datasize, model, weight, momentum, rounds=r)
        
        # aggregate the updates and update the server
        model, weight, momentum = server_opt(args, client_loader, client_datasize, model, weight, momentum, selected_clients, rounds=r)
        
        # update the history of selected_clients
        for sc in selected_clients: 
            selected_clients_num[sc] += 1
        
        # evaluate the generalization of the server model
        acc, std, min, max = evaluate_accuracy(model, dataset_sizes['test'], client_loader['test'], args)
        print('Test Accuracy: %.2f' % acc)
        
        test_acc.append(acc)
        if acc >= best_acc[0]: 
            best_acc = [acc, std, min, max]
            
        # record the results
        args.log_pd.loc[r] = [acc, std, min, max]
        args.log_pd.to_csv(args.log_filename)        

    # plot the results
    selected_clients_plotter(selected_clients_num, args)
    test_acc_plotter(test_acc, args)
    
    # save file
    state={}
    state['checkpoint'] = weight['server']
    torch.save(state, args.check_filename)
    
    print('Successfully saved' + check_filename)
    print('Best Test Accuracy: %.2f' % best_acc[0])
           
    args.log_pd.loc[args.num_rounds] = best_acc
    args.log_pd.to_csv(args.log_filename)

    
if __name__ == '__main__':
    train()
