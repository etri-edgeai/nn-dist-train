import os
import sys
import time
import logging
import collections
import csv

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

__all__ = ['load_federated_landmarks_g23k', 'load_federated_landmarks_g160k']

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
        

class Landmarks(data.Dataset):
    def __init__(self, data_dir, allfiles, dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False):
        """
        allfiles is [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ...  
                     {'user_id': xxx, 'image_id': xxx, 'class': xxx} ... ]
        """
        self.allfiles = allfiles
        if dataidxs == None:
            self.local_files = self.allfiles
        else:
            self.local_files = self.allfiles[dataidxs[0]: dataidxs[1]]
            # print("self.local_files: %d, dataidxs: (%d, %d)" % (len(self.local_files), dataidxs[0], dataidxs[1]))
        self.data_dir = data_dir
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # if self.user_id != None:
        #     return sum([len(local_data) for local_data in self.mapping_per_user.values()])
        # else:
        #     return len(self.mapping_per_user)
        return len(self.local_files)

    def __getitem__(self, idx):
        # if self.user_id != None:
        #     img_name = self.mapping_per_user[self.user_id][idx]['image_id']
        #     label = self.mapping_per_user[self.user_id][idx]['class']
        # else:
        #     img_name = self.mapping_per_user[idx]['image_id']
        #     label = self.mapping_per_user[idx]['class']
        img_name = self.local_files[idx]['image_id']
        label = int(self.local_files[idx]['class'])

        img_name = os.path.join(self.data_dir, str(img_name) + ".jpg")

        # convert jpg to PIL (jpg -> Tensor -> PIL)
        image = Image.open(img_name)
        # jpg_to_tensor = transforms.ToTensor()
        # tensor_to_pil = transforms.ToPILImage()
        # image = tensor_to_pil(jpg_to_tensor(image))
        # image = jpg_to_tensor(image)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

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
    
    
def _read_csv(path: str):  
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


def _data_transforms_landmarks():
    # IMAGENET_MEAN = [0.5071, 0.4865, 0.4409]
    # IMAGENET_STD = [0.2673, 0.2564, 0.2762]

    IMAGENET_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STD = [0.5, 0.5, 0.5]

    image_size = 224
    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, valid_transform



def _get_mapping_per_user(fn, args):
    """
    mapping_per_user is {'user_id': [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ... {}], 
                         'user_id': [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ... {}],
    } or               
                        [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ...  
                         {'user_id': xxx, 'image_id': xxx, 'class': xxx} ... ]
    }
    """
    mapping_table = _read_csv(fn)
    expected_cols = ['user_id', 'image_id', 'class']
    if not all(col in mapping_table[0].keys() for col in expected_cols):
        logger.error('%s has wrong format.', mapping_file)
        raise ValueError(
            'The mapping file must contain user_id, image_id and class columns. '
            'The existing columns are %s' % ','.join(mapping_table[0].keys()))

    data_local_num_dict = np.zeros(args.num_clients)


    mapping_per_user = collections.defaultdict(list)
    data_files = []
    net_dataidx_map = {}
    sum_temp = 0

    for row in mapping_table:
        user_id = row['user_id']
        mapping_per_user[user_id].append(row)
    for user_id, data in mapping_per_user.items():
        num_local = len(mapping_per_user[user_id])
        # net_dataidx_map[user_id]= (sum_temp, sum_temp+num_local)
        # data_local_num_dict[user_id] = num_local
        net_dataidx_map[int(user_id)]= (sum_temp, sum_temp+num_local)
        data_local_num_dict[int(user_id)] = num_local
        sum_temp += num_local
        data_files += mapping_per_user[user_id]
    assert sum_temp == len(data_files)

    return data_files, data_local_num_dict, net_dataidx_map


# for centralized training
def _get_dataloader(dataset, datadir, train_files, test_files, train_bs, test_bs, dataidxs=None):
    return _get_dataloader_Landmarks(datadir, train_files, test_files, train_bs, test_bs, dataidxs)


def _get_dataloader_Landmarks(datadir, train_files, test_files, train_bs, test_bs, dataidxs=None):
    dl_obj = Landmarks

    transform_train, transform_test = _data_transforms_landmarks()

    train_ds = dl_obj(datadir, train_files, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, test_files, dataidxs=None, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def load_federated_landmarks_g23k(args):
    fed_g23k_train_map_file = './data/landmark/data_user_dict/gld23k_user_dict_train.csv'
    fed_g23k_test_map_file = './data/landmark/data_user_dict/gld23k_user_dict_test.csv'

    if (not os.path.isfile(os.path.join(args.data_dir, fed_g23k_train_map_file))) or (not os.path.isfile(os.path.join(args.ata_dir, fed_g23k_test_map_file))):
        os.system('bash ./data/scripts/download_landmark.sh')
        
    client_number = 233
    fed_train_map_file = fed_g23k_train_map_file
    fed_test_map_file = fed_g23k_test_map_file
        
    train_files, data_local_num_dict, net_dataidx_map = _get_mapping_per_user(fed_train_map_file, args)
    test_files = _read_csv(fed_test_map_file)

    class_num = len(np.unique([item['class'] for item in train_files]))
    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = len(train_files)

    train_data_global, test_data_global = _get_dataloader(dataset, args.data_dir, train_files, test_files, args.batch_size, args.batch_size)
    # logging.info("train_dl_global number = " + str(len(train_data_global)))
    # logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_files)

    # get local dataset
    data_local_num_dict = data_local_num_dict
    train_data_local_dict = dict()
    test_data_local_dict = dict()


    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        # local_data_num = len(dataidxs)
        local_data_num = dataidxs[1] - dataidxs[0]
        # data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = _get_dataloader(dataset, args.data_dir, train_files, test_files, args.batch_size, args.batch_size,
                                                 dataidxs)
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    client_loader = {'train': train_data_local_dict, 'test': test_data_global}
    dataset_sizes = {'train': data_local_num_dict, 'test': test_data_num}
    
    return client_loader, dataset_sizes


def load_federated_landmarks_g160k(args):
    fed_g160k_train_map_file = './data/landmark/data_user_dict/gld160k_user_dict_train.csv'
    fed_g160k_map_file = './data/landmark/data_user_dict/gld160k_user_dict_test.csv'

    if (not os.isfile(os.path.join(args.data_dir, fed_g23k_train_map_file))) or (not os.isfile(os.path.join(args.data_dir, fed_g23k_test_map_file))):
        os.system('bash ./data/scripts/download_landmark.sh')

    client_number = 1262 
    fed_train_map_file = fed_g160k_train_map_file
    fed_test_map_file = fed_g160k_map_file
        
    train_files, data_local_num_dict, net_dataidx_map = _get_mapping_per_user(fed_train_map_file, args)
    test_files = _read_csv(fed_test_map_file)

    class_num = len(np.unique([item['class'] for item in train_files]))
    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = len(train_files)

    train_data_global, test_data_global = _get_dataloader(dataset, args.data_dir, train_files, test_files, args.batch_size, args.batch_size)
    # logging.info("train_dl_global number = " + str(len(train_data_global)))
    # logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_files)

    # get local dataset
    data_local_num_dict = data_local_num_dict
    train_data_local_dict = dict()
    test_data_local_dict = dict()


    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        # local_data_num = len(dataidxs)
        local_data_num = dataidxs[1] - dataidxs[0]
        # data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = _get_dataloader(dataset, args.data_dir, train_files, test_files, args.batch_size, args.batch_size,
                                                 dataidxs)
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    client_loader = {'train': train_data_local_dict, 'test': test_data_global}
    dataset_sizes = {'train': data_local_num_dict, 'test': test_data_num}
    
    return client_loader, dataset_sizes