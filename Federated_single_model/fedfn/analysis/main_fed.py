#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model, record_net_data_stats
from models.Update import *
from models.test import test_img_local, test_img_local_all, test_img_global
import os

import pdb
import sys
import random
import time

import logging


UPDATE = {
    "FedAvg": LocalUpdateAvg,
    "FedProx": LocalUpdateProx,
    "FedNTD": LocalUpdateNTD,
    "Scaffold": LocalUpdateScaffold,
    "FedLSD": LocalUpdateLSD,
    "FedMSE": LocalUpdateMSE,
    "FedNEW": LocalUpdateNEW
    
}




if __name__ == '__main__':
    
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # parse args
    args = args_parser()
    
    # Seed
    torch.manual_seed(args.seed)#args.running_idx=args.seed
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    assert args.fl_alg in ['FedAvg', 'FedProx', 'FedNTD', 'Scaffold', 'FedLSD', 'FedMSE', 'FedNEW']

    assert args.scheduler in ['multistep', 'step']
    
    assert args.local_upt_part in ['body', 'head', 'full'] 
    
    assert args.model in ['mobile', 'avg', 'cnn', 'resnet', 'shuffle', 'shufflenet', 'vgg']     
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    
    if args.iid:
        base_dir = './save/full_and_body/{}_iid{}_num{}_C{}_le{}_m{}_wd{}_round_{}/decay_{}/'.format(
            args.model, args.iid, args.num_users, args.frac, args.local_ep, args.momentum, args.wd, args.epochs, args.lr_decay)
        
    else:
        if args.hetero_option=="shard":
            base_dir = './save/full_and_body/{}_iid{}_num{}_C{}_le{}_m{}_wd{}_round_{}/shard{}/decay_{}/'.format(
                args.model, args.iid, args.num_users, args.frac, args.local_ep, args.momentum, args.wd, args.epochs, args.shard_per_user, args.lr_decay)


        elif args.hetero_option=="lda":
                base_dir = './save/full_and_body/{}_iid{}_num{}_C{}_le{}_m{}_wd{}_round_{}/alpha{}/decay_{}/'.format(
                    args.model, args.iid, args.num_users, args.frac, args.local_ep, args.momentum, args.wd, args.epochs,args.alpha, args.lr_decay)
                
            
    if args.fn:
        if args.fl_alg=="FedAvg" or args.fl_alg=="Scaffold":
            algo_dir = 'fn_{}/seed_{}/norm_{}/{}/local_upt_{}_lr_{}'.format(args.fn, args.seed, args.feature_norm, args.fl_alg ,args.local_upt_part, args.lr)
        elif args.fl_alg=="FedProx":
            algo_dir = 'fn_{}/seed_{}/norm_{}/{}/mu_{}/local_upt_{}_lr_{}'.format(args.fn, args.seed, args.feature_norm, args.fl_alg, args.mu, args.local_upt_part, args.lr)
        elif args.fl_alg=="FedNTD" or args.fl_alg=="FedLSD" or args.fl_alg=="FedMSE" or args.fl_alg=="FedNEW" :
            algo_dir = 'fn_{}/seed_{}/norm_{}/{}/ce_{}_kd_{}/local_upt_{}_lr_{}'.format(args.fn, args.seed, args.feature_norm, args.fl_alg, args.w_ce, args.w_kd, args.local_upt_part, args.lr)
            
    else:
        if args.fl_alg=="FedAvg" or args.fl_alg=="Scaffold":
            algo_dir = 'fn_{}/seed_{}/{}/local_upt_{}_lr_{}'.format(args.fn, args.seed, args.fl_alg ,args.local_upt_part, args.lr)
        elif args.fl_alg=="FedProx":
            algo_dir = 'fn_{}/seed_{}/{}/mu_{}/local_upt_{}_lr_{}'.format(args.fn, args.seed, args.fl_alg, args.mu, args.local_upt_part, args.lr)
        elif args.fl_alg=="FedNTD" or args.fl_alg=="FedLSD" or args.fl_alg=="FedMSE" or args.fl_alg=="FedNEW" :
            algo_dir = 'fn_{}/seed_{}/{}/ce_{}_kd_{}/local_upt_{}_lr_{}'.format(args.fn, args.seed, args.fl_alg, args.w_ce, args.w_kd, args.local_upt_part, args.lr)
        
        
    
    print("base_dir:", base_dir)
    print("algo_dir:", algo_dir)
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)#exist_ok=True: 디렉토리가 없으면 새로 만들고 없으면 아무 일 없는거임!!
        
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    
    
    if args.iid:
        dict_save_path = 'dict_users_10.pkl'
    else:
        if args.hetero_option=="shard":
            dict_save_path = 'dict_users_10_{}.pkl'.format(args.shard_per_user)
        elif args.hetero_option=="lda":
            dict_save_path = 'dict_users_lda_{}.pkl'.format(args.alpha)
    
    with open(dict_save_path, 'rb') as handle:#기존 pretrained되었을 때 쓰였던 클라이언트 구성으로 덮어씌운다.
        dict_users_train, dict_users_test = pickle.load(handle)



    print(">>> Distributing client train data...")
    traindata_cls_dict = record_net_data_stats(dict_users_train, np.array(dataset_train.targets))
    logging.info('Data statistics: %s' % str(traindata_cls_dict))
    
    print(">>> Distributing client test data...")
    testdata_cls_dict = record_net_data_stats(dict_users_test, np.array(dataset_test.targets))
    logging.info('Data statistics: %s' % str(testdata_cls_dict))
    
    
    unq, unq_cnt = np.unique(np.array(dataset_train.targets), return_counts=True)
    
    args.num_classes=len(unq)
    
    print(args)
    
    # build a global model
    net_glob = get_model(args) #global model 그자체!!
    net_glob.train()
    print(net_glob)

    # build local models
    net_global_storage = []
    net_global_storage.append(copy.deepcopy(net_glob))
    
    # training
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    
    norm_stat, similarity_stat={}, {}
    head_params = [p for name, p in net_glob.named_parameters() if 'classifier' in name]
    
    norm_stat[0]=torch.diagonal(torch.mm(head_params[0],head_params[0].transpose(0,1))).cpu().detach().numpy()
    
    normalized_classifier=nn.functional.normalize(head_params[0], p=2, dim=1)
    
    similarity_stat[0]=torch.mm(normalized_classifier, normalized_classifier.transpose(0,1)).cpu().detach().numpy()
    
    for iter in range(args.epochs):
        
        w_glob = None
        loss_locals = []
        data_size=[]
        
        # Client Sampling
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))

        # Local Updates
        
        for idx in idxs_users:
            local = UPDATE[args.fl_alg](args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = copy.deepcopy(net_global_storage[0])
            if args.local_upt_part == 'body':
                w_local, loss, train_size = local.train(net=net_local.to(args.device), body_lr=lr, head_lr=0.)
            if args.local_upt_part == 'head':
                w_local, loss, train_size = local.train(net=net_local.to(args.device), body_lr=0., head_lr=lr)
            if args.local_upt_part == 'full':
                w_local, loss, train_size = local.train(net=net_local.to(args.device), body_lr=lr, head_lr=lr)
                
            loss_locals.append(copy.deepcopy(loss))
            data_size.append(train_size)

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)

            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
        
        # Aggregation
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)
        
        
        # Broadcast
        net_global_storage[0].load_state_dict(w_glob, strict=True)
        
        head_params = [p for name, p in net_global_storage[0].named_parameters() if 'classifier' in name]
    
        norm_stat[iter + 1]=torch.diagonal(torch.mm(head_params[0],head_params[0].transpose(0,1))).cpu().detach().numpy()
    
        normalized_classifier=nn.functional.normalize(head_params[0], p=2, dim=1)
    
        similarity_stat[iter + 1]=torch.mm(normalized_classifier, normalized_classifier.transpose(0,1)).cpu().detach().numpy()
    
            
        if args.scheduler=='multistep':
            if (iter + 1) in [args.epochs//2, (args.epochs*3)//4]:
                lr *= args.lr_decay
        elif args.scheduler=='step':
            lr*=args.lr_decay

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)#selected된 클라이언트들의 whole train dataset기준 averaged된 loss  기재!!
        
        

        if (iter + 1) % args.test_freq == 0:
            if args.hetero_option=="lda":
                acc_test, loss_test = test_img_global(net_global_storage[0], dataset_test, args)
            else: 
                acc_test, loss_test = test_img_local_all(net_global_storage, args, dataset_test, dict_users_test, return_all=False)
                        
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter
                
#                 best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
                
#                 torch.save(net_global_storage[0].state_dict(), best_save_path)
                
            results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
#             final_results.to_csv(results_save_path, index=False)
            
        if (iter + 1) in [args.epochs//2, (args.epochs*3)//4]:
            middle_save_path = os.path.join(base_dir, algo_dir, 'middle_model.pt')

            torch.save(net_global_storage[0].state_dict(), middle_save_path)

            sys.exit()
            
#     stat_path = os.path.join(base_dir, algo_dir, 'stat.pkl')
#     with open(stat_path, 'wb') as handle:
#         pickle.dump((norm_stat, similarity_stat), handle)
            

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc)) #finetuning은 아직 안한 상태!!