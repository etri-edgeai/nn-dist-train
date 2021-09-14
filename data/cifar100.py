import logging
import os

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

__all__ = ['load_federated_cifar100']


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

client_ids_train = None
client_ids_test = None
DEFAULT_TRAIN_CLIENTS_NUM = 500
DEFAULT_TEST_CLIENTS_NUM = 100
DEFAULT_TRAIN_FILE = 'fed_cifar100_train.h5'
DEFAULT_TEST_FILE = 'fed_cifar100_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_IMGAE = 'image'
_LABEL = 'label'


def _cifar100_transform(img_mean, img_std, train = True, crop_size = (32,32)):
    """cropping, flipping, and normalizing."""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])


def _preprocess_cifar_img(img, train):
    # scale img to range [0,1] to fit ToTensor api
    img = torch.div(img, 255.0)
    transoformed_img = torch.stack([_cifar100_transform
        (i.type(torch.DoubleTensor).mean(),
            i.type(torch.DoubleTensor).std(),
            train)
        (i.permute(2,0,1)) 
        for i in img])
    return transoformed_img


def _get_dataloader(args, client_idx=None):
    train_bs, test_bs = args.batch_size, args.batch_size
    
    train_h5 = h5py.File(os.path.join(args.data_dir, DEFAULT_TRAIN_FILE), 'r')
    test_h5 = h5py.File(os.path.join(args.data_dir, DEFAULT_TEST_FILE), 'r')
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    
    # load data in numpy format from h5 file
    if client_idx is None:
        train_x = np.vstack([train_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in client_ids_train])
        train_y = np.vstack([train_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in client_ids_train]).squeeze()
        test_x = np.vstack([test_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in client_ids_test])
        test_y = np.vstack([test_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in client_ids_test]).squeeze()
    else:
        client_id_train = client_ids_train[client_idx]
        train_x = np.vstack([train_h5[_EXAMPLE][client_id_train][_IMGAE][()]])
        train_y = np.vstack([train_h5[_EXAMPLE][client_id_train][_LABEL][()]]).squeeze()
        if client_idx <= len(client_ids_test) - 1:
            client_id_test = client_ids_test[client_idx]
            test_x = np.vstack([train_h5[_EXAMPLE][client_id_test][_IMGAE][()]])
            test_y = np.vstack([train_h5[_EXAMPLE][client_id_test][_LABEL][()]]).squeeze()

    # preprocess 
    train_x = _preprocess_cifar_img(torch.tensor(train_x), train=True)
    train_y = torch.tensor(train_y)
    if len(test_x) != 0:
        test_x = _preprocess_cifar_img(torch.tensor(test_x), train=False)
        test_y = torch.tensor(test_y)
    
    # generate dataloader
    train_ds = data.TensorDataset(train_x, train_y)
    train_dl = data.DataLoader(dataset=train_ds,
                               batch_size=train_bs,
                               shuffle=True,
                               drop_last=False)

    if len(test_x) != 0:
        test_ds = data.TensorDataset(test_x, test_y)
        test_dl = data.DataLoader(dataset=test_ds,
                                  batch_size=test_bs,
                                  shuffle=True,
                                  drop_last=False)
    else:
        test_dl = None

    train_h5.close()
    test_h5.close()
    return train_dl, test_dl


def load_federated_cifar100(args):
    args.data_dir = os.path.join(args.data_dir, 'federated_cifar100')
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
        
    if (not os.path.isfile(os.path.join(args.data_dir, DEFAULT_TRAIN_FILE))) or (not os.path.isfile(os.path.join(args.data_dir, DEFAULT_TEST_FILE))):
        os.system('bash ./data/scripts/download_cifar100.sh')
        
    #client id list
    train_file_path = os.path.join(args.data_dir, DEFAULT_TRAIN_FILE)
    test_file_path = os.path.join(args.data_dir, DEFAULT_TEST_FILE)
    with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
        global client_ids_train, client_ids_test
        client_ids_train = list(train_h5[_EXAMPLE].keys())
        client_ids_test = list(test_h5[_EXAMPLE].keys())
    
    args.num_clients = DEFAULT_TRAIN_CLIENTS_NUM
    
    # get local dataset
    data_local_num_dict = np.zeros(DEFAULT_TRAIN_CLIENTS_NUM)
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    
    for client_idx in range(DEFAULT_TRAIN_CLIENTS_NUM):
        train_data_local, test_data_local = _get_dataloader(args, client_idx)
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        logging.info("client_idx = %d, batch_num_train_local = %d" % (client_idx, len(train_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    
    # global dataset
    train_data_global = data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(train_data_local_dict.values()))
                ),
                batch_size=args.batch_size, shuffle=True)
    train_data_num = len(train_data_global.dataset)
    
    test_data_global = data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(test_data_local_dict.values()) if dl is not None)
                ),
                batch_size=args.batch_size, shuffle=True)
    test_data_num = len(test_data_global.dataset)
    
    args.in_channels = 3
    args.num_classes = 100

    client_loader = {'train': train_data_local_dict, 'test': test_data_global}
    dataset_sizes = {'train': data_local_num_dict, 'test': test_data_num}
    
    return client_loader, dataset_sizes, args