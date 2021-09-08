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
from utils.util import fix_seed, set_path, gpu_to_cpu, cpu_to_gpu, save_checkpoint
from utils.plotter import test_acc_plotter, selected_clients_plotter
from data import load_dirichlet_data, load_femnist, load_federated_cifar100, load_federated_landmarks_g23k, load_federated_landmarks_g160k, load_federated_synthetic
from models import *
from train_tools import client_opt
from train_tools.server_opt import server_opt

DATASET = {'dirichlet_cifar10': load_dirichlet_data, 'dirichlet_cifar100': load_dirichlet_data, 'dirichlet_mnist': load_dirichlet_data, 'dirichlet_fmnist': load_dirichlet_data, 'femnist': load_federated_emnist, 'cifar100': load_federated_cifar100, 'landmark_g23k': load_federated_landmarks_g23k, 'landmark_g160k': load_federated_landmarks_g160k, 'synthetic': load_federated_synthetic}

MODEL = {'lenet': LeNet, 'lenet_container': LeNetContainer, 'resnet8': resnet8, 'vgg11': vgg11, 'vgg11_bn': vgg11_bn, 'vgg13': vgg13, 'vgg13_bn': vgg13_bn, 'vgg16': vgg16, 'vgg16_bn': vgg16_bn, 'vgg19': vgg19, 'vgg19_bn': vgg19_bn}


def _get_args():
    # get argument for training
    args = parse_args()
    
    # make experiment reproducible
    fix_seed(args.seed)
        
    # model log file
    args, log_pd = set_path(args)
    
    return args, log_pd
    
    
def _make_model(args):
    # create model for server and client
    model = MODEL[args.model](num_classes=args.num_classes, **args.model_kwargs) if args.model_kwargs else MODEL[args.model](num_classes=args.num_classes)
        
    # initialize server and client model weights
    server_weight = copy.deepcopy(model.state_dict())
    server_momentum = {}
    
    client_weight = {}
    client_momentum = {}
    for client in range(args.num_clients):
        client_weight[client] = copy.deepcopy(model.state_dict())
        client_momentum[client] = {}
    
    weight = {'server': server_weight,
              'client': client_weight}
    momentum = {'server': server_momentum,
                'client': client_momentum}
    
    # model to gpu
    model = model.to(args.device)
    
    return model, weight, momentum

    
def train():
    args, log_pd = _get_args()
    
    # create dataloader
    client_loader, dataset_sizes = DATASET[args.dataset](args) 
    
    model, model, weight, momentum = _make_model(args)

    # evaluation metrics
    selected_clients_num = np.zeros(args.num_clients)
    best_acc = [-np.inf, 0, 0, 0]
    test_acc = []
    
    # Federated Learning Pipeline
    for r in tqdm(range(args.num_rounds)):
        # client selection and updated the selected clients
        weight, momentum, selected_clients = client_opt(args, client_loader, dataset_sizes['train'], model, weight, momentum, r)
        
        # aggregate the updates and update the server
        weight, momentum, model = server_opt(args, client_loader, dataset_sizes['train'], model, weight, momentum, selected_clients, rounds=r)
        
        # update the history of selected_clients
        for sc in selected_clients: 
            selected_clients_num[sc] += 1
        
        # evaluate the generalization of the server model
        mean, std, min, max = test(args, model, client_loader, dataset_sizes)
        
        test_acc.append(mean)
        if mean >= best_acc[0]: 
            best_acc = [mean, std, min, max]
            
        # record the results
        log_pd.loc[r] = [mean, std, min, max]
        log_pd.to_csv(args.log_path)        

    # plot the results
    test_acc_plotter(args, test_acc)
    selected_clients_plotter(args, selected_clients_num)
    
    # save checkpoint
    save_checkpoint(args, weight['server'])
           
    log_pd.loc[args.num_rounds] = best_acc
    log_pd.to_csv(args.log_path)
    print('Best Test Accuracy: %.2f' % best_acc[0])

    
def test(args, model, client_loader, dataset_sizes):
    prediction = lambda logits : torch.argmax(logits, 1)
    accuracy = np.zeros(args.num_classes)

    model.eval()
    for inputs, labels in client_loader['test']:
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.set_grad_enabled(False):
            logits = model(inputs)
            pred = prediction(logits)

            correct = (pred == labels.data)
            for cls in labels[correct]:
                accuracy[cls] += 1.0

    accuracy = (accuracy / (dataset_sizes['test'] / args.num_classes)) * 100
    mean = round((np.sum(accuracy) / args.num_classes), 2)
    std = round(np.std(accuracy), 4)
    min = round(np.min(accuracy), 2)
    max = round(np.max(accuracy), 2)

    print('Test Accuracy: %.2f' % mean)
    
    return mean, std, min, max

    
if __name__ == '__main__':
    train()
