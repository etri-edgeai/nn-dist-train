import logging
import logging
import math

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import CIFAR10_truncated
from ..augmentation import RandAugment
from fedml_api.utils.utils_condense import get_dataset

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# generate the non-IID distribution for all methodsƒ
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def _data_transforms_noaug_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)
 
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def load_cifar10_data_noaug(datadir):
    train_transform, test_transform = _data_transforms_noaug_cifar10()

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)
 
    return cifar10_train_ds, cifar10_test_ds
    

def partition_data(dataset, datadir, partition, n_nets, alpha, valid_ratio=0.0):
    logging.info("*********partition data***************")
    X_train_all, y_train_all, X_test, y_test = load_cifar10_data(datadir)
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

def partition_data_equally(dataset, datadir, partition, n_nets, alpha, valid_ratio=0.0, train_ratio=1.0):
    logging.info("*********partition data equally***************")
    n_auxi_nets = 10
    X_train_all, y_train_all, X_test, y_test = load_cifar10_data(datadir)
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

        # X_train = X_train_all[train_idxs]
        # y_train = y_train_all[train_idxs]
        
        train_idxs = train_idxs[:int(train_ratio * len(train_idxs))]
        X_train = X_train_all[train_idxs]
        y_train = y_train_all[train_idxs]
        
        for subset_idx, original_idx in enumerate(train_idxs):
            subset2original_idx[subset_idx] = original_idx 
            original2subset_idx[original_idx] = subset_idx
    else :
        train_idxs = total_idxs
        train_idxs = total_idxs[:int(train_ratio * len(train_idxs))]

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

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

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


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_unlabeled_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None, num_workers=2, randaug=False):
    # For ensemble distillation, shuffle off + return num of train and test -> Why shuffle off?

    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    if randaug:
        transform_train.transforms.insert(3, RandAugment(3,5)) # Need to check the order where the Randaug is inserted

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_data_num = len(train_ds)
    test_data_num = len(test_ds)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True,
                               drop_last=False, num_workers=num_workers, pin_memory=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,
                             drop_last=False)

    return train_data_num, test_data_num, train_dl, test_dl


def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None, num_workers=2, randaug=False):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()
    if randaug:
        transform_train.transforms.insert(3, RandAugment(3,5)) # Need to check the order where the Randaug is inserted


    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True,
                               drop_last=False, num_workers=num_workers, pin_memory=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,
                              drop_last=False)

    return train_dl, test_dl

def get_dataloader_val_CIFAR10(datadir, train_bs, test_bs, dataidxs=None, num_workers=0):
    # Test transforms for validation set

    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_test, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False,
                               drop_last=False, num_workers=num_workers, pin_memory=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,
                              drop_last=False)

    return train_dl, test_dl

def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None,
                                num_workers=2):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False,
                               num_workers=num_workers)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False,
                              num_workers=num_workers)

    return train_dl, test_dl


def load_partition_data_distributed_cifar10(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, \
           train_data_local, test_data_local, class_num


def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number,
                                batch_size, valid_ratio=0.0, split_equally=False, randaug=False, condense=False, train_ratio=1.0):
    

    '''For condense'''
    if condense :
        data_local_noaug = dict()
       # _, _, _, _, _, _, dst_train_noaug, _, _ = get_dataset('CIFAR10', data_dir)
        dst_train_noaug, _ = load_cifar10_data_noaug(data_dir)
        #x_train, y_train, x_test, y_test = load_cifar10_data(data_dir)
        

    
    if split_equally :
        partitioned_data = partition_data_equally(dataset,
                                                data_dir,
                                                partition_method,
                                                client_number,
                                                partition_alpha,
                                                valid_ratio,
                                                train_ratio)

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
        dataidxs = valid_idxs[0]
        # validation batch size 1024 for fast validation and 2 num_workers
        valid_data_global, _ = get_dataloader_val_CIFAR10(data_dir, 1024, 64, dataidxs, num_workers=0)
        return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, valid_data_global
    
    elif valid_ratio > 0.0 and condense:
        # Get valid dataloader
        dataidxs = valid_idxs[0]
        # validation batch size 1024 for fast validation and 2 num_workers
        valid_data_global, _ = get_dataloader_val_CIFAR10(data_dir, 1024, 64, dataidxs, num_workers=0)
        
        return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, valid_data_global, data_local_noaug
    
    
    if condense :
        return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, data_local_noaug
        
        

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


# def load_partition_data_cifar10_validation(dataset, data_dir, partition_method, partition_alpha, client_number,
#                                            batch_size, valid_ratio=0.0):
#     X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
#                                                                                              data_dir,
#                                                                                              partition_method,
#                                                                                              client_number,
#                                                                                              partition_alpha, valid_ratio)
#     class_num = len(np.unique(y_train))
#     logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
#     train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

#     train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
#     logging.info("train_dl_global number = " + str(len(train_data_global)))
#     logging.info("test_dl_global number = " + str(len(test_data_global)))
#     test_data_num = len(test_data_global)

#     # get local dataset
#     data_local_num_dict = dict()
#     train_data_local_dict = dict()
#     test_data_local_dict = dict()

#     for client_idx in range(client_number):
#         dataidxs = net_dataidx_map[client_idx]
#         local_data_num = len(dataidxs)
#         data_local_num_dict[client_idx] = local_data_num
#         logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

#         # training batch size = 64; algorithms batch size = 32
#         train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
#                                                  dataidxs)
#         logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
#             client_idx, len(train_data_local), len(test_data_local)))
#         train_data_local_dict[client_idx] = train_data_local
#         test_data_local_dict[client_idx] = test_data_local
#     return train_data_num, test_data_num, train_data_global, test_data_global, \
#            data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
