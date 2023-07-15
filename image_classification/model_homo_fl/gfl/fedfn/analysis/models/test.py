#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pdb
import sys

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
    
def test_img_local(net_g, dataset, args, user_idx=-1, idxs=None, return_features=False):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(dataset, batch_size=args.bs)
    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.bs, shuffle=False)
    l = len(data_loader)
        
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        
        if return_features:
            tmp_features = net_g.extract_features(data)
            
            if idx==0:
                features = tmp_features.detach().cpu()
                targets = target.detach().cpu()
            else:
                features = torch.cat([features, tmp_features.detach().cpu()], dim=0)
                targets = torch.cat([targets, target.detach().cpu()])

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
#     if args.verbose:
#         print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
#             user_idx, test_loss, correct, len(data_loader.dataset), accuracy))
        
    if return_features:
        return accuracy, test_loss, features, targets
    else:
        return accuracy, test_loss
    
def test_img_local_all(net_global_storage, args, dataset_test, dict_users_test, return_all=False):
    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    for idx in range(args.num_users):
        net_local = net_global_storage[0]
        net_local.eval()
        a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])

        acc_test_local[idx] = a#local test accuracy
        loss_test_local[idx] = b#local test loss
    data_ratio_local = np.zeros(args.num_users)
    for idx in range(args.num_users):
        idxs = dict_users_test[idx]
        data_ratio_local[idx] = len(DatasetSplit(dataset_test, idxs)) / len(dataset_test)

    if return_all:#select된 모든 client들의 test data에서의 acc, loss return
        return acc_test_local, loss_test_local
#     return acc_test_local.mean(), loss_test_local.mean()
    return (acc_test_local*data_ratio_local).sum(), (loss_test_local*data_ratio_local).sum()#select된 모든 client들의 averagede된 test data에서의 acc, loss return

def test_img_global(net_g, dataset, args, user_idx=-1, idxs=None, return_features=False):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(dataset, batch_size=args.bs)
    data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=False)
    l = len(data_loader)
        
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        
        if return_features:
            tmp_features = net_g.extract_features(data)
            
            if idx==0:
                features = tmp_features.detach().cpu()
                targets = target.detach().cpu()
            else:
                features = torch.cat([features, tmp_features.detach().cpu()], dim=0)
                targets = torch.cat([targets, target.detach().cpu()])

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
#     if args.verbose:
#         print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
#             user_idx, test_loss, correct, len(data_loader.dataset), accuracy))
        
    if return_features:
        return accuracy, test_loss, features, targets
    else:
        return accuracy, test_loss


