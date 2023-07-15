#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
from itertools import permutations
import numpy as np
import torch
import pdb

def fair_iid(dataset, num_users):
    """
    Sample I.I.D. client data from fairness dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fair_noniid(train_data, num_users, num_shards=200, num_imgs=300, train=True, rand_set_all=[]):
    """
    Sample non-I.I.D client data from fairness dataset
    :param dataset:
    :param num_users:
    :return:
    """
    assert num_shards % num_users == 0
    shard_per_user = int(num_shards / num_users)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)

    #import pdb; pdb.set_trace()

    labels = train_data[1].numpy().reshape(len(train_data[0]),)
    assert num_shards * num_imgs == len(labels)
    #import pdb; pdb.set_trace()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    if len(rand_set_all) == 0:
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
            for rand in rand_set:
                rand_set_all.append(rand)

            idx_shard = list(set(idx_shard) - rand_set) # remove shards from possible choices for other users
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    else: # this only works if the train and test set have the same distribution of labels
        for i in range(num_users):
            rand_set = rand_set_all[i*shard_per_user: (i+1)*shard_per_user]
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    return dict_users, rand_set_all

def iid(dataset, num_users, server_data_ratio):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    if server_data_ratio > 0.0:
        dict_users['server'] = set(np.random.choice(all_idxs, int(len(dataset)*server_data_ratio), replace=False))
    
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    return dict_users

def noniid(dataset, num_users, shard_per_user, server_data_ratio, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users, all_idxs = {i: np.array([], dtype='int64') for i in range(num_users)}, [i for i in range(len(dataset))]
    
    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))

    if server_data_ratio > 0.0:
        dict_users['server'] = set(np.random.choice(all_idxs, int(len(dataset)*server_data_ratio), replace=False))
    
    return dict_users, rand_set_all

def noniid_replace(dataset, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for key, value in dict_users.items():
        assert(len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return dict_users, rand_set_all


def iid_unbalanced(dataset, num_users, num_batch_users, moved_data_size):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    assert moved_data_size // 5 == moved_data_size / 5
    assert (num_users//2) // num_batch_users == (num_users//2) / num_batch_users
    
    if len(dataset) == 10000:
        moved_data_size = moved_data_size // 5 
    
    avg_num_items = int(len(dataset)/num_users)
    
    num_items = [avg_num_items] * num_users
    num_items = np.array(num_items)

    num_step = int(num_users/num_batch_users)
    moved_num_items = np.zeros([num_step, num_batch_users], dtype=int)

    for i in range(len(moved_num_items)):
        moved_num_items[i, :] = moved_data_size * (i+1)

    assert (int(np.mean(moved_num_items))==np.mean(moved_num_items))

    moved_num_items = moved_num_items.flatten()
    moved_num_items = moved_num_items - int(np.mean(moved_num_items))
    num_items = (num_items + moved_num_items).tolist()
    print (num_items)
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid_unbalanced(dataset, num_users, num_batch_users, moved_data_size, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # Step 1) Calculate the number of data allocated for each client
    assert moved_data_size // 5 == moved_data_size / 5
    assert (num_users//2) // num_batch_users == (num_users//2) / num_batch_users
    
    if len(dataset) == 10000:
        moved_data_size = moved_data_size // 5 
    
    avg_num_items = int(len(dataset)/num_users)
    
    num_items = [avg_num_items] * num_users
    num_items = np.array(num_items)

    num_step = int(num_users/num_batch_users)
    moved_num_items = np.zeros([num_step, num_batch_users], dtype=int)

    for i in range(len(moved_num_items)):
        moved_num_items[i, :] = moved_data_size * (i+1)

    assert (int(np.mean(moved_num_items))==np.mean(moved_num_items))

    moved_num_items = moved_num_items.flatten()
    moved_num_items = moved_num_items - int(np.mean(moved_num_items))
    num_items = (num_items + moved_num_items).tolist()
    print (num_items)
    
    # Step 2) Data allocation for the label assigned by the client pair unit
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    
    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    
    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * int(shard_per_class / 2)
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((int(num_users / 2), -1))
        
    # divide and assign
    for i in range(int(num_users / 2)):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            index_lst = np.random.choice(len(idxs_dict[label]), int(num_items[i] / shard_per_user), replace=False)
            
            idx_lst = [idx for index, idx in enumerate(idxs_dict[label]) if index in index_lst]
            idxs_dict[label] = list(set(idxs_dict[label]) - set(idx_lst))
            
            rand_set.append(idx_lst)
        dict_users[i] = np.concatenate(rand_set)
        
        pair_i = (num_users - 1) - i
        rand_set = []
        for label in rand_set_label:
            index_lst = np.random.choice(len(idxs_dict[label]), int(num_items[pair_i] / shard_per_user), replace=False)
            
            idx_lst = [idx for index, idx in enumerate(idxs_dict[label]) if index in index_lst]
            idxs_dict[label] = list(set(idxs_dict[label]) - set(idx_lst))
            
            rand_set.append(idx_lst)
            
        dict_users[pair_i] = np.concatenate(rand_set)
    
    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))
    
    return dict_users, rand_set_all