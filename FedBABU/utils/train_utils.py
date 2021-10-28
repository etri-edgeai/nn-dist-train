from torchvision import datasets, transforms
from models.Nets import CNNCifar, MobileNetCifar
from models.ResNet import ResNet18, ResNet50
from utils.sampling import iid, noniid, iid_unbalanced, noniid_unbalanced

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def get_data(args, env='fed'):
    if env == 'single':
        if args.dataset == 'cifar10':
            dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
            dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
            
        elif args.dataset == 'cifar100':
            dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
            dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        return dataset_train, dataset_test
    
    elif env == 'fed':
        if args.unbalanced:
            if args.dataset == 'cifar10':
                dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
                dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
                if args.iid:
                    dict_users_train = iid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size)
                    dict_users_test = iid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size)
                else:
                    dict_users_train, rand_set_all = noniid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user)
                    dict_users_test, rand_set_all = noniid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar100':
                dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
                dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
                if args.iid:
                    dict_users_train = iid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size)
                    dict_users_test = iid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size)
                else:
                    dict_users_train, rand_set_all = noniid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user)
                    dict_users_test, rand_set_all = noniid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user, rand_set_all=rand_set_all)
            else:
                exit('Error: unrecognized dataset')

        else:
            if args.dataset == 'mnist':
                dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
                dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
                # sample users
                if args.iid:
                    dict_users_train = iid(dataset_train, args.num_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, args.num_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar10':
                dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
                dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
                if args.iid:
                    dict_users_train = iid(dataset_train, args.num_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, args.num_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar100':
                dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
                dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
                if args.iid:
                    dict_users_train = iid(dataset_train, args.num_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, args.num_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            else:
                exit('Error: unrecognized dataset')

        return dataset_train, dataset_test, dict_users_train, dict_users_test

