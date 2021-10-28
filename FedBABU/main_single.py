#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.options import args_parser
from utils.train_utils import get_data, get_model

if __name__ == '__main__':
    # parse args
    args = args_parser()
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    base_dir = './save/{}/{}_single_{}/{}/'.format(args.dataset, args.model, args.opt, args.results_save)
    algo_dir = 'blr_{}_hlr{}_bm{}_hm_{}'.format(args.body_lr, args.head_lr, args.body_m, args.head_m)
    
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    # set dataset
    dataset_train, dataset_test = get_data(args, env='single')
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=128, num_workers=4)
    dataloaders = {'train': train_loader,
                   'test': test_loader}
    
    # build a model
    net_glob = get_model(args)
    
    # Basically, He uniform
    if args.results_save=='xavier_uniform':
        nn.init.xavier_uniform_(net_glob.linear.weight, gain=nn.init.calculate_gain('relu'))
    elif args.results_save=='xavier_normal':
        nn.init.xavier_normal_(net_glob.linear.weight, gain=nn.init.calculate_gain('relu'))
    elif args.results_save=='kaiming_uniform':
        nn.init.kaiming_uniform_(net_glob.linear.weight, nonlinearity='relu')
    elif args.results_save=='kaiming_normal':
        nn.init.kaiming_normal(net_glob.linear.weight, nonlinearity='relu')
    elif args.results_save=='orthogonal':
        nn.init.orthogonal_(net_glob.linear.weight, gain=nn.init.calculate_gain('relu'))
    elif args.results_save=='not_orthogonal':
        nn.init.uniform_(net_glob.linear.weight, a=0.45, b=0.55)
        net_glob.linear.weight.data = net_glob.linear.weight.data / torch.norm(net_glob.linear.weight.data, dim=1, keepdim=True)
    
    nn.init.zeros_(net_glob.linear.bias)
        
    # set optimizer
    body_params = [p for name, p in net_glob.named_parameters() if not 'linear' in name]
    head_params = [p for name, p in net_glob.named_parameters() if 'linear' in name]
    
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': args.body_lr, 'momentum': args.body_m},
                                     {'params': head_params, 'lr': args.head_lr, 'momentum': args.head_m}],
                                     weight_decay=5e-4)
        
    elif args.opt == 'RMSProp':
        optimizer = torch.optim.RMSprop([{'params': body_params, 'lr': args.body_lr, 'momentum': args.body_m},
                                         {'params': head_params, 'lr': args.head_lr, 'momentum': args.head_m}],
                                          weight_decay=5e-4)
    elif args.opt == 'ADAM':
        optimizer = torch.optim.Adam([{'params': body_params, 'lr': args.body_lr, 'betas': (args.body_m, 1.11*args.body_m)},
                                      {'params': head_params, 'lr': args.head_lr, 'betas': (args.head_m, 1.11*args.head_m)}],
                                       weight_decay=5e-4)
    
    # set scheduler    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     [80, 120],
                                                     gamma=0.1,
                                                     last_epoch=-1)
    
    
    # set criterion
    criterion = nn.CrossEntropyLoss()
    
    # training
    results_log_save_path = os.path.join(base_dir, algo_dir, 'results.csv')
    results_model_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
            
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in tqdm(range(args.epochs)):
        net_glob.train()
        train_loss = 0
        train_correct = 0
        train_data_num = 0
        
        for i, data in enumerate(dataloaders['train']):
            image = data[0].type(torch.FloatTensor).to(args.device)
            label = data[1].type(torch.LongTensor).to(args.device)
            
            pred_label = net_glob(image)
            loss = criterion(pred_label, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_label = torch.argmax(pred_label, dim=1)
            train_loss += loss.item()
            train_correct += (torch.sum(pred_label==label).item())
            train_data_num += label.shape[0]
        
        net_glob.eval()
        test_loss = 0
        test_correct = 0
        test_data_num = 0
        for i, data in enumerate(dataloaders['test']):
            image = data[0].type(torch.FloatTensor).to(args.device)
            label = data[1].type(torch.LongTensor).to(args.device)

            pred_label = net_glob(image)                
            loss = criterion(pred_label, label)
            
            pred_label = torch.argmax(pred_label, dim=1)
            test_loss += loss.item()
            test_correct += (torch.sum(pred_label==label).item())
            test_data_num += label.shape[0]
            
        train_loss_list.append(train_loss/len(dataloaders['train']))
        train_acc_list.append(train_correct/train_data_num)
        test_loss_list.append(test_loss/len(dataloaders['test']))
        test_acc_list.append(test_correct/test_data_num)
        
        res_pd = pd.DataFrame(data=np.array([train_loss_list, train_acc_list, test_loss_list, test_acc_list]).T,
                              columns=['train_loss', 'train_acc', 'test_loss', 'test_acc'])
        res_pd.to_csv(results_log_save_path, index=False)
        if (test_correct/test_data_num) >= max(test_acc_list):
            torch.save(net_glob.state_dict(), results_model_save_path)
            
        scheduler.step()