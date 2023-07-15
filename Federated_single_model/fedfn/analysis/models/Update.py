#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import math
import pdb
import copy
from torch.optim import Optimizer
import sys
import time

import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from criterion import NTD_Loss, LSD_Loss, MSE_Loss, NEW_Loss

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdateAvg(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.train_size=len(idxs)


    def train(self, net, body_lr, head_lr):
        net.train()

        # train and update
        
        # For ablation study
        """
        body_params = []
        head_params = []
        for name, p in net.named_parameters():
            if 'features.0' in name or 'features.1' in name: # active
                body_params.append(p)
            else: # deactive
                head_params.append(p)
        """
        body_params = [p for name, p in net.named_parameters() if 'classifier' not in name]
        head_params = [p for name, p in net.named_parameters() if 'classifier' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        
        

        epoch_loss = []


        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)
                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.train_size
        
        
    
    
    
# class LocalUpdateAvg(object):
#     def __init__(self, args, dataset=None, idxs=None):
#         self.args = args
#         self.loss_func = nn.CrossEntropyLoss()
#         self.selected_clients = []
#         self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

#         self.train_size=len(idxs)


#     def train(self, net, body_lr, head_lr):
#         net.train()

#         # train and update
        
#         # For ablation study
#         """
#         body_params = []
#         head_params = []
#         for name, p in net.named_parameters():
#             if 'features.0' in name or 'features.1' in name: # active
#                 body_params.append(p)
#             else: # deactive
#                 head_params.append(p)
#         """
#         body_params = [p for name, p in net.named_parameters() if 'classifier' not in name]
#         head_params = [p for name, p in net.named_parameters() if 'classifier' in name]
        
#         optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
#                                      {'params': head_params, 'lr': head_lr}],
#                                     momentum=self.args.momentum,
#                                     weight_decay=self.args.wd)
        
        
#         if self.train_size>130:

#             epoch_loss = []


#             for iter in range(self.args.local_ep):
#                 batch_loss = []
#                 for batch_idx, (images, labels) in enumerate(self.ldr_train):

#                     images, labels = images.to(self.args.device), labels.to(self.args.device)
#                     net.zero_grad()
#                     logits = net(images)
#                     loss = self.loss_func(logits, labels)
#                     loss.backward()
#                     optimizer.step()

#                     batch_loss.append(loss.item())

#                 epoch_loss.append(sum(batch_loss)/len(batch_loss))

#             return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.train_size
        
        
#         else:

#             self.iterations=3*self.args.local_ep

#             batch_loss = []

#             for iter in range(self.iterations):
#                 for batch_idx, (images, labels) in enumerate(self.ldr_train):
#                     images, labels = images.to(self.args.device), labels.to(self.args.device)

#                     net.zero_grad()
#                     logits = net(images)
#                     loss = self.loss_func(logits, labels)
#                     loss.backward()
#                     optimizer.step()

#                     batch_loss.append(loss.item())

#                     if batch_idx==0:
#                         break


#             return net.state_dict(), sum(batch_loss) / len(batch_loss), self.train_size    
    

# class LocalUpdateAvg(object):
#     def __init__(self, args, dataset=None, idxs=None):
#         self.args = args
#         self.loss_func = nn.CrossEntropyLoss()
#         self.selected_clients = []
#         self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

#         self.train_size=len(idxs)
#         self.iterations=10*self.args.local_ep


#     def train(self, net, body_lr, head_lr):
#         net.train()

#         # train and update
        
#         # For ablation study
#         """
#         body_params = []
#         head_params = []
#         for name, p in net.named_parameters():
#             if 'features.0' in name or 'features.1' in name: # active
#                 body_params.append(p)
#             else: # deactive
#                 head_params.append(p)
#         """
#         body_params = [p for name, p in net.named_parameters() if 'classifier' not in name]
#         head_params = [p for name, p in net.named_parameters() if 'classifier' in name]
        
#         optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
#                                      {'params': head_params, 'lr': head_lr}],
#                                     momentum=self.args.momentum,
#                                     weight_decay=self.args.wd)

#         epoch_loss = []
        
#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.ldr_train):

#                 images, labels = images.to(self.args.device), labels.to(self.args.device)
#                 net.zero_grad()
#                 logits = net(images)
#                 loss = self.loss_func(logits, labels)
#                 loss.backward()
#                 optimizer.step()

#                 batch_loss.append(loss.item())

#             epoch_loss.append(sum(batch_loss)/len(batch_loss))

#         return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.train_size


# #         batch_loss = []
        

# #         for iter in range(self.iterations):
# #             for batch_idx, (images, labels) in enumerate(self.ldr_train):
# #                 images, labels = images.to(self.args.device), labels.to(self.args.device)
# #                 import ipdb; ipdb.set_trace(context=15)
            
# #                 net.zero_grad()
# #                 logits = net(images)
# #                 loss = self.loss_func(logits, labels)
# #                 loss.backward()
# #                 optimizer.step()

# #                 batch_loss.append(loss.item())
                
# #                 if batch_idx==0:
# #                     break
                    

# #         return net.state_dict(), sum(batch_loss) / len(batch_loss), self.train_size




class LocalUpdateProx(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        
        self.train_size=len(idxs)

    def train(self, net, body_lr, head_lr):
        net.train()
        g_net = copy.deepcopy(net)
        
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                
                # for fedprox
                fed_prox_reg = 0.0
                for l_param, g_param in zip(net.parameters(), g_net.parameters()):
                    fed_prox_reg += (self.args.mu / 2 * torch.norm((l_param - g_param)) ** 2)
                loss += fed_prox_reg
                
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.train_size 
    
    
class LocalUpdateNTD(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = NTD_Loss(num_classes=args.num_classes, tau=args.tau, w_ce=args.w_ce, w_kd=args.w_kd)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.train_size=len(idxs)
    
    

    def train(self, net, body_lr, head_lr):        
        self.teacher=copy.deepcopy(net)
        
        net.train()
        self.teacher.eval()

        # train and update
        
        # For ablation study
        """
        body_params = []
        head_params = []
        for name, p in net.named_parameters():
            if 'features.0' in name or 'features.1' in name: # active
                body_params.append(p)
            else: # deactive
                head_params.append(p)
        """
        body_params = [p for name, p in net.named_parameters() if 'classifier' not in name]
        head_params = [p for name, p in net.named_parameters() if 'classifier' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)
                dg_logits=self.teacher(images)
                loss = self.loss_func(logits, labels, dg_logits)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.train_size


class LocalUpdateLSD(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = LSD_Loss(num_classes=args.num_classes, tau=args.tau, w_ce=args.w_ce, w_kd=args.w_kd)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.train_size=len(idxs)
    
    

    def train(self, net, body_lr, head_lr):
        self.teacher=copy.deepcopy(net)
        
        net.train()
        self.teacher.eval()

        # train and update
        
        # For ablation study
        """
        body_params = []
        head_params = []
        for name, p in net.named_parameters():
            if 'features.0' in name or 'features.1' in name: # active
                body_params.append(p)
            else: # deactive
                head_params.append(p)
        """
        body_params = [p for name, p in net.named_parameters() if 'classifier' not in name]
        head_params = [p for name, p in net.named_parameters() if 'classifier' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)
                dg_logits=self.teacher(images)
                loss = self.loss_func(logits, labels, dg_logits)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.train_size
    
class LocalUpdateMSE(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = MSE_Loss(num_classes=args.num_classes, tau=args.tau, w_ce=args.w_ce, w_kd=args.w_kd)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.train_size=len(idxs)
    
    

    def train(self, net, body_lr, head_lr):
        self.teacher=copy.deepcopy(net)
        
        net.train()
        self.teacher.eval()

        # train and update
        
        # For ablation study
        """
        body_params = []
        head_params = []
        for name, p in net.named_parameters():
            if 'features.0' in name or 'features.1' in name: # active
                body_params.append(p)
            else: # deactive
                head_params.append(p)
        """
        body_params = [p for name, p in net.named_parameters() if 'classifier' not in name]
        head_params = [p for name, p in net.named_parameters() if 'classifier' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)
                dg_logits=self.teacher(images)
                loss = self.loss_func(logits, labels, dg_logits)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.train_size
    
    

class LocalUpdateNEW(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = NEW_Loss(num_classes=args.num_classes, tau=args.tau, w_ce=args.w_ce, w_kd=args.w_kd)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.train_size=len(idxs)
    
    

    def train(self, net, body_lr, head_lr):
        self.teacher=copy.deepcopy(net)
        
        net.train()
        self.teacher.eval()

        # train and update
        
        # For ablation study
        """
        body_params = []
        head_params = []
        for name, p in net.named_parameters():
            if 'features.0' in name or 'features.1' in name: # active
                body_params.append(p)
            else: # deactive
                head_params.append(p)
        """
        body_params = [p for name, p in net.named_parameters() if 'classifier' not in name]
        head_params = [p for name, p in net.named_parameters() if 'classifier' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)
                dg_logits=self.teacher(images)
                loss = self.loss_func(logits, labels, dg_logits)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.train_size
    
    
    
    
class LocalUpdateScaffold(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = NTD_Loss(num_classes=100)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.train_size=len(idxs)
    
    

    def train(self, net, body_lr, head_lr):
        net.train()

        # train and update
        
        # For ablation study
        """
        body_params = []
        head_params = []
        for name, p in net.named_parameters():
            if 'features.0' in name or 'features.1' in name: # active
                body_params.append(p)
            else: # deactive
                head_params.append(p)
        """
        body_params = [p for name, p in net.named_parameters() if 'classifier' not in name]
        head_params = [p for name, p in net.named_parameters() if 'classifier' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)
                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.train_size

    
    
    
