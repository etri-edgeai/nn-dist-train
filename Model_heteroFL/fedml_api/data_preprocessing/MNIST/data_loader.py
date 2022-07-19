import json
import logging

import os

import math
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torchvision.transforms as transforms


# def read_data(train_data_dir, test_data_dir):
#     '''parses data in given train and test data directories

#     assumes:
#     - the data in the input directories are .json files with 
#         keys 'users' and 'user_data'
#     - the set of train set users is the same as the set of test set users

#     Return:
#         clients: list of non-unique client ids
#         groups: list of group ids; empty list if none found
#         train_data: dictionary of train data
#         test_data: dictionary of test data
#     '''
#     clients = []
#     groups = []
#     train_data = {}
#     test_data = {}

#     train_files = os.listdir(train_data_dir)
#     train_files = [f for f in train_files if f.endswith('.json')]
#     for f in train_files:
#         file_path = os.path.join(train_data_dir, f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         clients.extend(cdata['users'])
#         if 'hierarchies' in cdata:
#             groups.extend(cdata['hierarchies'])
#         train_data.update(cdata['user_data'])

#     test_files = os.listdir(test_data_dir)
#     test_files = [f for f in test_files if f.endswith('.json')]
#     for f in test_files:
#         file_path = os.path.join(test_data_dir, f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         test_data.update(cdata['user_data'])

#     clients = sorted(cdata['users'])

#     return clients, groups, train_data, test_data


# def batch_data(data, batch_size):
#     '''
#     data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
#     returns x, y, which are both numpy array of length: batch_size
#     '''
#     data_x = data['x']
#     data_y = data['y']

#     # randomly shuffle data
#     np.random.seed(100)
#     rng_state = np.random.get_state()
#     np.random.shuffle(data_x)
#     np.random.set_state(rng_state)
#     np.random.shuffle(data_y)

#     # loop through mini-batches
#     batch_data = list()
#     for i in range(0, len(data_x), batch_size):
#         batched_x = data_x[i:i + batch_size]
#         batched_y = data_y[i:i + batch_size]
#         batched_x = torch.from_numpy(np.asarray(batched_x)).float()
#         batched_y = torch.from_numpy(np.asarray(batched_y)).long()
#         batch_data.append((batched_x, batched_y))
#     return batch_data


# def load_partition_data_mnist_by_device_id(batch_size,
#                                            device_id,
#                                            train_path="MNIST_mobile",
#                                            test_path="MNIST_mobile"):
#     train_path += '/' + device_id + '/' + 'train'
#     test_path += '/' + device_id + '/' + 'test'
#     return load_partition_data_mnist(batch_size, train_path, test_path)


# def load_partition_data_mnist(batch_size,
#                               train_path="./../../../data/MNIST/train",
#                               test_path="./../../../data/MNIST/test"):
#     users, groups, train_data, test_data = read_data(train_path, test_path)

#     if len(groups) == 0:
#         groups = [None for _ in users]
#     train_data_num = 0
#     test_data_num = 0
#     train_data_local_dict = dict()
#     test_data_local_dict = dict()
#     train_data_local_num_dict = dict()
#     train_data_global = list()
#     test_data_global = list()
#     client_idx = 0
#     for u, g in zip(users, groups):
#         user_train_data_num = len(train_data[u]['x'])
#         user_test_data_num = len(test_data[u]['x'])
#         train_data_num += user_train_data_num
#         test_data_num += user_test_data_num
#         train_data_local_num_dict[client_idx] = user_train_data_num

#         # transform to batches
#         train_batch = batch_data(train_data[u], batch_size)
#         test_batch = batch_data(test_data[u], batch_size)

#         # index using client index
#         train_data_local_dict[client_idx] = train_batch
#         test_data_local_dict[client_idx] = test_batch
#         train_data_global += train_batch
#         test_data_global += test_batch
#         client_idx += 1
#     client_num = client_idx
#     class_num = 10

#     return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
#            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
def get_dataset(data_dir, train=True):

    trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    if train:
        dataset = datasets.MNIST(data_dir, train=True, download=True, transform=trans_mnist)
    else :
        dataset = datasets.MNIST(data_dir, train=False, download=True, transform=trans_mnist)
    
    return dataset
    
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, num_workers=2):
    
    train_ds = get_dataset(datadir, train=True)
    test_ds = get_dataset(datadir, train=False)
    
    if dataidxs :
        train_dl = DataLoader(DatasetSplit(train_ds, dataidxs), batch_size=train_bs, shuffle=True,
              drop_last=False, num_workers=num_workers, pin_memory=True)
    else :
        train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True,
              drop_last=False, num_workers=num_workers, pin_memory=True )
    test_dl = DataLoader(test_ds, batch_size=test_bs, shuffle=False,
              drop_last=False, num_workers=num_workers, pin_memory=True)
    
    return train_dl, test_dl

def get_dataloader_val(datadir, train_bs, test_bs, dataidxs=None, num_workers=2):
    # Same transform for Train and val
    
    train_ds = get_dataset(datadir, train=True)
    test_ds = get_dataset(datadir, train=False)
    
    if type(dataidxs) == list :
        dataidxs = dataidxs[0]
    
    val_dl = DataLoader(DatasetSplit(train_ds, dataidxs), batch_size=train_bs, shuffle=True,
              drop_last=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=test_bs, shuffle=False,
              drop_last=False, num_workers=num_workers, pin_memory=True)
    
    return val_dl, test_dl
    
    

def load_mnist_data(data_dir):
    
    trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    
    dataset_train = datasets.MNIST(data_dir, train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST(data_dir, train=False, download=True, transform=trans_mnist)
    
    X_train, y_train = dataset_train.data, dataset_train.targets
    X_test, y_test= dataset_test.data, dataset_test.targets
    
    return X_train, y_train, X_test, y_test
 
    
def load_mnist_data_noaug(data_dir):
    "MNIST has no extra augmentation so its identical with the original one"
    return get_dataset(data_dir, train=True)
    
def load_partition_data_mnist(dataset, data_dir, partition_method, partition_alpha, client_number,
                                batch_size, valid_ratio=0.0, split_equally=False, randaug=False, condense=False):
    
    '''For condense'''
    if condense :
        data_local_noaug = dict()
        #_, _, _, _, _, _, dst_train_noaug, _, _ = get_dataset('CIFAR10', data_dir)
        dst_train_noaug  = load_mnist_data_noaug(data_dir)
    
    if split_equally :
        partitioned_data = partition_data_equally(dataset,
                                                data_dir,
                                                partition_method,
                                                client_number,
                                                partition_alpha,
                                                valid_ratio)

    else :
        partitioned_data = partition_data(dataset,
                                       data_dir,
                                       partition_method,
                                       client_number,
                                       partition_alpha,
                                       valid_ratio)

    if valid_ratio > 0.0 :
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, *valid_idxs = partitioned_data
    else :
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partitioned_data

    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    
    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        
        # For dataset condense
        if condense :
            ''' load no aug dataset for all clients '''
            dataidxs = net_dataidx_map[client_idx]
            images_all = [torch.unsqueeze(dst_train_noaug[idx][0], dim=0) for idx in dataidxs] 
            labels_all = [dst_train_noaug[idx][1] for idx in dataidxs]
            images_all = torch.cat(images_all, dim=0)
            labels_all = torch.tensor(labels_all, dtype=torch.long)
            data_local_noaug[client_idx] = (images_all, labels_all)

    if valid_ratio > 0.0 and not condense:
        # Get valid dataloader
        dataidxs = valid_idxs
        # validation batch size 1024 for fast validation and 2 num_workers
        valid_data_global, _ = get_dataloader_val(data_dir, 1024, 64, dataidxs, num_workers=0)

        return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, valid_data_global
    
    elif valid_ratio > 0.0 and condense:
        # Get valid dataloader
        dataidxs = valid_idxs
        # validation batch size 1024 for fast validation and 2 num_workers
        valid_data_global, _ = get_dataloader_val(data_dir, 1024, 64, dataidxs, num_workers=0)
        
        return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, valid_data_global, data_local_noaug
    

    if condense :
        return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, data_local_noaug
        
        

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num



def partition_data_equally(dataset, datadir, partition, n_nets, alpha, valid_ratio=0.0):
    logging.info("*********partition data equally***************")
    n_auxi_nets = 10
    X_train_all, y_train_all, X_test, y_test = load_mnist_data(datadir)
    n_train = X_train_all.shape[0]
    # n_test = X_test.shape[0]
    total_idxs = np.random.permutation(n_train)

    X_valid = None
    y_valid = None
    valid_idxs = None
    subset2original_idx = {}
    original2subset_idx = {}
    
    if valid_ratio > 0.0:
        valid_n = int(valid_ratio * n_train)
        train_idxs = total_idxs[valid_n:]
        valid_idxs = total_idxs[:valid_n]
        X_valid = X_train_all[valid_idxs]
        y_valid = y_train_all[valid_idxs]

        X_train = X_train_all[train_idxs]
        y_train = y_train_all[train_idxs]
        
        for subset_idx, original_idx in enumerate(train_idxs):
            subset2original_idx[subset_idx] = original_idx 
            original2subset_idx[original_idx] = subset_idx
    else :
        train_idxs = total_idxs
        X_train = X_train[train_idxs]
        y_train = y_train[train_idxs]

    if partition == 'homo':
        total_num = X_train.shape[0]
        train_idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(train_idxs, n_nets)
        net_dataidx_map = {i : batch_idxs[i] for i in range(n_nets)}

    elif partition == 'hetero':
        assert n_auxi_nets <= n_nets
        # Divide indicies to smaller groups

        K = 10 ### Number of class  !
        net_dataidx_map = {}
        num_indicies = X_train.shape[0]
        y_indicies = [i for i in range(num_indicies)]
        from_index = 0
        splited_y_train = []
        splited_y_indicies = []

        num_splits = math.ceil(n_nets / n_auxi_nets)

        split_n_nets = [n_auxi_nets
                            if idx < num_splits - 1
                            else n_nets - n_auxi_nets * (num_splits-1)
                            for idx in range(num_splits)]
        split_ratios = [_n_nets / n_nets for _n_nets in split_n_nets]

        for idx, ratio in enumerate(split_n_nets):
            to_index = from_index + int(n_auxi_nets / n_nets * num_indicies)

            splited_y_train.append(
                y_train[
                from_index : (num_indicies if idx == num_splits - 1 else to_index)
                                    ]
            )

            splited_y_indicies.append(
                y_indicies[
                    from_index : (num_indicies if idx == num_splits - 1 else to_index)
                ]
            )
            from_index = to_index

        idx_batch = []
        tmp_n_nets = n_nets
        for _y_indices, _y_train in zip(splited_y_indicies, splited_y_train):
            _y_indices = np.array(_y_indices)
            _y_train = np.array(_y_train)
            _y_train_size = len(_y_train)

            # Use auxi nets for this subset targets
            _n_nets = min(n_auxi_nets, tmp_n_nets)
            tmp_n_nets = tmp_n_nets - n_auxi_nets

            min_size = 0
            while min_size < int(0.50 * _y_train_size / _n_nets):
                _idx_batch = [[] for _ in range(_n_nets)]
                for k in range(K):
                    idx_tmp = np.where(_y_train == k)[0]
                    idx_k = _y_indices[idx_tmp]
                    np.random.shuffle(idx_k)

                    propotions = np.random.dirichlet(np.repeat(alpha, _n_nets))
                    ## Balance
                    propotions = np.array([p * (len(idx_j) < _y_train_size / _n_nets) for p, idx_j in zip(propotions, _idx_batch)])
                    propotions = propotions / propotions.sum()
                    propotions = (np.cumsum(propotions) * len(idx_k)).astype(int)[:-1]
                    _idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(_idx_batch, np.split(idx_k, propotions))]
                    min_size = min([len(idx_j) for idx_j in _idx_batch])

            for j in range(_n_nets):
                np.random.shuffle(_idx_batch[j])
            idx_batch += _idx_batch

        for j in range(n_nets):
            net_dataidx_map[j] =idx_batch[j]

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)


    if valid_ratio :
        # Change subset index to original index
        net_dataidx_map_all = {}
        for k,v in net_dataidx_map.items():
            index_all = [subset2original_idx[index] for index in v]
            net_dataidx_map_all[k] = index_all
            
        return X_train, y_train, X_test, y_test, net_dataidx_map_all, traindata_cls_counts, valid_idxs

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

def partition_data(dataset, datadir, partition, n_nets, alpha, valid_ratio=0.0):
    logging.info("*********partition data***************")
    X_train_all, y_train_all, X_test, y_test = load_mnist_data(datadir)
    n_train = X_train_all.shape[0]
    # n_test = X_test.shape[0]
    total_idxs = np.random.permutation(n_train)

    X_valid = None
    y_valid = None
    valid_idxs = None

    if valid_ratio > 0.0:
        valid_n = int(valid_ratio * n_train)
        train_idxs = total_idxs[valid_n:]
        valid_idxs = total_idxs[:valid_n]
        X_valid = X_train_all[valid_idxs]
        y_valid = y_train_all[valid_idxs]

        X_train = X_train_all[train_idxs]
        y_train = y_train_all[train_idxs]
    else :
        train_idxs = total_idxs
        X_train = X_train_all[train_idxs]
        y_train = y_train_all[train_idxs]

    if partition == "homo":
        total_num = X_train
        train_idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(train_idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
        #while min_size < int(0.50 * N / n_nets):
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    if valid_ratio:
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, valid_idxs

    return  X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts