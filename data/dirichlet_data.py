import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, sampler, Subset
from utils import clients_data_num_plotter

__all__ = ['load_federated_dirichlet_data']


def _get_mean_std(dataset):
    if dataset == 'dirichlet_cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset == 'dirichlet_cifar100':
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
    elif dataset == 'dirichlet_mnist':
        mean = [0.5]
        std = [0.5]
    elif dataset == 'dirichlet_fmnist':
        mean = [0.5]
        std = [0.5]
    else:
        raise Exception('No option for this dataset.')

    return mean, std


def _transform_setter(dataset=str):
    mean, std = _get_mean_std(dataset=dataset)
    
    # train, test augmentation
    if dataset == 'dirichlet_mnist' or dataset == 'dirichlet_fmnist':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return train_transforms, test_transforms


def _divide_dataset(args, _trainset, num_classes=10):
    # Here, we handle the non-iidness via the Dirichlet distribution.
    # non_iid: the concentration parameter alpha in the Dirichlet distribution. Should be float type.
    # We refer to the paper 'https://arxiv.org/pdf/1909.06335.pdf'
    num_clients = args.num_clients
    valid_size = args.valid_size
    non_iid = args.non_iid
    
    assert type(num_clients) is int, 'num_clients should be the type of integer.'
    assert type(valid_size) is int, 'valid_size should be the type of int.'
    assert type(non_iid) is float and non_iid > 0, 'iid should be the type of float.'
    
    # Generate the set of clients and valid dataset.
    clients_data = {}
    for i in range(num_clients):
        clients_data[i] = []
    clients_data['valid'] = []

    # Divide the dataset into each class of dataset.
    total_data = {}
    for i in range(num_classes):
        total_data[str(i)] = []
    for idx, data in enumerate(_trainset):
        total_data[str(data[1])].append(idx)
    
    # Generate the valid dataset.
    for cls in total_data.keys():
        tmp =  random.sample(total_data[cls], valid_size)
        total_data[cls] = list(set(total_data[cls]) - set(tmp))
        clients_data['valid'] += tmp

    clients_data_num = {}
    for client in range(num_clients):
        clients_data_num[client] = [0] * num_classes
    
    # Distribute the data with the Dirichilet distribution.
    diri_dis = torch.distributions.dirichlet.Dirichlet(non_iid * torch.ones(num_classes))
    remain = np.inf
    nums = int((len(_trainset)-num_classes*valid_size) / num_clients)
    while remain != 0:
        for client_idx in clients_data.keys():
            if client_idx != 'valid':
                if len(clients_data[client_idx]) >= nums:
                    continue

                tmp = diri_dis.sample()
                for cls in total_data.keys():
                    tmp_set = random.sample(total_data[cls], min(len(total_data[cls]), int(nums * tmp[int(cls)] / 15)))

                    if len(clients_data[client_idx]) + len(tmp_set) > nums:
                        tmp_set = tmp_set[:nums-len(clients_data[client_idx])]
                    clients_data[client_idx] += tmp_set

                    clients_data_num[client_idx][int(cls)] += len(tmp_set)
                    total_data[cls] = list(set(total_data[cls])-set(tmp_set))   

        remain = sum([len(d) for _, d in total_data.items()])

    print('clients_data_num', [sum(clients_data_num[k]) for k in clients_data_num.keys()])
    
    # plot the data distribution
    clients_data_num_plotter(args, clients_data_num)

    return clients_data


def load_federated_dirichlet_data(args):
    root = args.data_dir
    if not os.path.isdir(root):
        os.makedirs(root)
        
    train_transforms, test_transforms = _transform_setter(dataset=args.dataset)
    if args.dataset == 'dirichlet_cifar10':
        _trainset = datasets.CIFAR10(os.path.join(root, args.dataset), train=True, transform = train_transforms, download = True)
        testset = datasets.CIFAR10(os.path.join(root, args.dataset), train=False, transform = test_transforms, download = False)
        num_classes = 10
        
    elif args.dataset == 'dirichlet_cifar100':
        _trainset = datasets.CIFAR100(os.path.join(root, args.dataset), train=True, transform = train_transforms, download = True)
        testset = datasets.CIFAR100(os.path.join(root, args.dataset), train=False, transform = test_transforms, download = False)
        num_classes = 100
        
    elif args.dataset == 'dirichlet_mnist':
        _trainset = datasets.MNIST(os.path.join(root, args.dataset), train=True, transform = train_transforms, download = True)
        testset = datasets.MNIST(os.path.join(root, args.dataset), train=False, transform = test_transforms, download = False)
        num_classes = 10
        
    elif args.dataset == 'dirichlet_fmnist':
        _trainset = datasets.FashionMNIST(os.path.join(root, args.dataset), train=True, transform = train_transforms, download = True)
        testset = datasets.FashionMNIST(os.path.join(root, args.dataset), train=False, transform = test_transforms, download = False)
        num_classes = 10

    args.num_classes = num_classes
    clients_data = _divide_dataset(args, _trainset, num_classes=num_classes)
    
    # Generate the dataloader
    client_loader = {'train': {}}
    dataset_sizes = {'train': np.zeros(args.num_clients)}
    for client_idx in clients_data.keys():
        subset = Subset(_trainset, clients_data[client_idx])
        
        if client_idx == 'valid':
            client_loader['valid'] = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)
            dataset_sizes['valid'] = len(clients_data[client_idx])
            
        else:
            client_loader['train'][client_idx] = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers)
            dataset_sizes['train'][client_idx] = len(clients_data[client_idx])
                        
    client_loader['test'] = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataset_sizes['test'] = len(testset)
    
    # client_loader: data loader of each client. type is dictionary
    # dataset_sizes: the number of data for each client. type is dictionary
    return client_loader, dataset_sizes, args