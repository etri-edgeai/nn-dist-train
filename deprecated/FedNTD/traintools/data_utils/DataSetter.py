import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from collections import defaultdict, deque
from termcolor import colored
import numpy as np
import os, random

from .FEMNIST import *

__all__ = ["DataSetter"]

DATACLASS = {
    "mnist": datasets.MNIST,
    "fashion-mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}

DATASIZE = {
    "mnist": 28,
    "femnist": 28,
    "fashion-mnist": 28,
    "cifar10": 32,
    "cifar100": 32,
}

CLASSNUM = {
    "mnist": 10,
    "femnist": 62,
    "fashion-mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
}

DATASTAT = {
    "mnist": {"mean": [0.1307], "std": [0.3081]},
    "femnist": {"mean": [0.9618], "std": [0.1644]},
    "fashion-mnist": {"mean": [0.5], "std": [0.5]},
    "cifar10": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
    "cifar100": {"mean": [0.5071, 0.4865, 0.4409], "std": [0.1980, 0.2010, 0.1970]},
}


class DataSetter:
    """
    Assigns data to local clients.
    """

    def __init__(self, root="./data", dataset="cifar10"):
        self.root = root
        self.data_root = os.path.join(root, dataset)
        self.dataset = dataset
        self.num_classes = CLASSNUM[dataset]

        if dataset not in ["femnist"]:
            self.default_transform = {
                "train": transforms.Compose(
                    [
                        transforms.RandomCrop(
                            DATASIZE[dataset], padding=DATASIZE[dataset] // 8
                        ),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=DATASTAT[dataset]["mean"], std=DATASTAT[dataset]["std"]
                        ),
                    ]
                ),
                "test": transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=DATASTAT[dataset]["mean"], std=DATASTAT[dataset]["std"]
                        ),
                    ]
                ),
            }

    def data_distributer(
        self,
        n_clients=100,
        alg="fedavg",
        max_class_num=2,
        dir_alpha=0.5,
        train_transform=None,
        test_transform=None,
        as_dict=True,
    ):
        """distribute train data to clients."""

        # Pre-defined heterogeneity setting with 100 clients for FEMNIST datasets.
        if self.dataset == "femnist":
            trainset = FEMNIST(self.data_root, split="train")
            testset = FEMNIST(self.data_root, split="test")

            local_trainset = []

            start = 0
            for i in range(100):
                user_samples = trainset.num_samples[i]
                indicies = range(len(trainset.labels))[start : start + user_samples]
                local_trainset.append(Subset(trainset, indicies))
                start += user_samples

            return {"local": local_trainset, "test": testset, "unlabeled": None}

        trainset, testset = self._dataset_getter(
            train_transform, test_transform
        )

        # get heterogeneous data distribution
        locals_idx = self._data_processor(
            trainset, n_clients, alg, max_class_num, dir_alpha
        )

        # distribute data to clients
        local_trainset = []
        for i in range(n_clients):
            local_idx = locals_idx.pop()
            local_trainset.append(Subset(trainset, local_idx))

        if as_dict:
            return {"local": local_trainset, "test": testset}

        return local_trainset, testset, unlabeledset

    def _dataset_getter(self, train_transform=None, test_transform=None):
        """Make datasets to build"""

        if train_transform is None:
            train_transform = self.default_transform["train"]

        if test_transform is None:
            test_transform = self.default_transform["test"]

        trainset = DATACLASS[self.dataset](
            self.data_root, train=True, transform=train_transform, download=True
        )
        testset = DATACLASS[self.dataset](
            self.data_root, train=False, transform=test_transform, download=True
        )

        return trainset, testset

    def _data_processor(
        self, trainset, n_clients=100, alg="fedavg", max_class_num=2, dir_alpha=0.5
    ):
        """get heterogeneous data distribution"""
        np.random.seed(2021)
        random.seed(2021)
        
        labels = trainset.targets
        length = int(len(labels) / n_clients)
        idx = []

        # homogeneous distribution (i.i.d. setting)
        if alg == "iid" or n_clients == 1:
            tot_idx = np.arange(len(labels))
            for _ in range(n_clients):
                idx.append(tot_idx[:length])
                tot_idx = tot_idx[length:]

            return deque(idx)

        else:
            # make data shard and distribute
            shard_size = int(length / max_class_num)  # e.g. 300 = 600 / 2
            unique_classes = np.unique(labels)

            tot_idx_by_label = []  # shape: class x num_shards x shard_size
            for i in unique_classes:
                idx_by_label = np.where(labels == i)[0]
                tmp = []
                while 1:
                    tmp.append(idx_by_label[:shard_size])
                    idx_by_label = idx_by_label[shard_size:]
                    if len(idx_by_label) < shard_size / 2:
                        break
                tot_idx_by_label.append(tmp)

            if alg == "fedavg":
                # randomly select classes for each client
                for _ in range(n_clients):
                    idx_by_devices = []
                    while len(idx_by_devices) < max_class_num:
                        chosen_label = np.random.choice(
                            unique_classes, 1, replace=False
                        )[
                            0
                        ]  # 임의의 Label을 하나 뽑음
                        if (
                            len(tot_idx_by_label[chosen_label]) > 0
                        ):  # 만약 해당 Label의 shard가 하나라도 남아있다면,
                            l_idx = np.random.choice(
                                len(tot_idx_by_label[chosen_label]), 1, replace=False
                            )[
                                0
                            ]  # shard 중 일부를 하나 뽑고
                            idx_by_devices.append(
                                tot_idx_by_label[chosen_label][l_idx].tolist()
                            )  # 클라이언트에 넣어준다.
                            del tot_idx_by_label[chosen_label][
                                l_idx
                            ]  # 뽑힌 shard의 원본은 제거!
                    idx.append(np.concatenate(idx_by_devices))

            elif alg == "fedma":
                idx_batch = [[] for _ in range(n_clients)]
                idx = [defaultdict(list) for _ in range(n_clients)]
                for it, k in enumerate(unique_classes):
                    this_labels = np.concatenate(tot_idx_by_label[it])
                    prop = np.random.dirichlet([dir_alpha for _ in range(n_clients)])
                    prop = np.array(
                        [p * (len(idx_j) < length) for p, idx_j in zip(prop, idx_batch)]
                    )
                    prop = prop / prop.sum()
                    prop = (prop * len(this_labels)).astype(int).cumsum()[:-1]
                    label_by_device = np.split(this_labels, prop)
                    for device_id, lb in enumerate(label_by_device):
                        idx_batch[device_id] += lb.copy().tolist()
                        idx[device_id][k] = lb.copy().tolist()

                # show example client data distributions
                print(colored(f'{"Tot":5s}', "red"), end="")
                for i in range(10):
                    print(f"{i:5d}", end="")
                print("\n")

                for i in range(n_clients):
                    print(colored(f"{len(idx_batch[i]):5d}", "red"), end="")
                    for k in idx[i].keys():
                        print(f"{len(idx[i][k]):5d}", end="")
                    print("\n")

                idx = idx_batch

            else:
                raise RuntimeError

        return deque(idx)
