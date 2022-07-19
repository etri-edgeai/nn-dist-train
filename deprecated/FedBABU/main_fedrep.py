#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdateFedRep
from models.test import test_img, test_img_local, test_img_local_all
from models.Fed import FedAvg
import os

import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.unbalanced:
        base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}_unbalanced_bu{}_md{}/{}/'.format(
            args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.num_batch_users, args.moved_data_size, args.results_save)
    else:
        base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/'.format(
            args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.results_save)
    algo_dir = "fedrep"
    
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    dict_save_path = os.path.join(base_dir, algo_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    # build a global model
    net_glob = get_model(args)
    net_glob.train()

    # build local models
    net_local_list = []
    for user_idx in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))
    
    # training
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    w_glob = None
    
    update_keys = [k for k in net_glob.state_dict().keys() if 'linear' not in k]
    print("all keys", net_glob.state_dict().keys())
    print("aggregation keys", update_keys)
    w_glob = {k: net_glob.state_dict()[k] for k in update_keys}
    
    for iter in range(args.epochs):
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        w_locals = []

        # local updates
        for idx in idxs_users:
            net_local = copy.deepcopy(net_local_list[idx])
            
            # Server sends current representation φ^t to these clients
            net_local.load_state_dict(w_glob, strict=False)
        
            local = LocalUpdateFedRep(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            w = local.train(net=copy.deepcopy(net_local).to(args.device), lr=lr)
            w_locals.append(copy.deepcopy(w))
            net_local_list[idx].load_state_dict(copy.deepcopy(w))
            
        if (iter + 1) in [args.epochs//2, (args.epochs*3)//4]:
            lr *= 0.1
            
        # update global weights
        w_glob = FedAvg(w_locals)
        w_glob = {k: w_glob[k] for k in update_keys}
        net_glob.load_state_dict(w_glob, strict=False)
        
        
        # - Evaluation - #
        
        # Backup local parameters
        backup_net_local_list = copy.deepcopy(net_local_list)
        backup_local_epoch = args.local_ep
        
        # fine-tuning
        for idx in range(args.num_users):
            net_local = copy.deepcopy(net_local_list[idx])
            
            # Server sends current representation φ^t to these clients
            net_local.load_state_dict(w_glob, strict=False)
        
            local = LocalUpdateFedRep(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            w = local.train(net=copy.deepcopy(net_local).to(args.device), lr=lr)
            w_locals.append(copy.deepcopy(w))
            net_local_list[idx].load_state_dict(copy.deepcopy(w))
        
        if (iter + 1) % args.test_freq == 0:
            acc_test, loss_test = test_img_local_all(net_local_list, args, dataset_test, dict_users_test, return_all=False)
            
            print('Round {:3d}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_test, acc_test))

            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter
                
                for user_idx in range(args.num_users):
                    best_save_path = os.path.join(base_dir, algo_dir, 'best_local_{}.pt'.format(user_idx))
                    torch.save(net_local_list[user_idx].state_dict(), best_save_path)

            results.append(np.array([iter, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_test', 'acc_test', 'best_acc'])
            final_results.to_csv(results_save_path, index=False)
        
        # rollback local parameters
        net_local_list = copy.deepcopy(backup_net_local_list)

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))