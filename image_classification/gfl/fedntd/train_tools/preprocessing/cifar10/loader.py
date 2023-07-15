import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import sys, os

from .datasets import CIFAR10_truncated


class Cutout:
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

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),]
    )

    return train_transform, valid_transform


def get_all_targets_cifar10(root, train=True):
    dataset = CIFAR10_truncated(root=root, train=train)
    all_targets = dataset.targets
    return all_targets


# def get_all_targets_cifar5(root, train=True):
#     all_targets = get_all_targets_cifar10(root, train)
#     cifar5_targets = all_targets[all_targets < 5]
#     return cifar5_targets


def get_dataloader_cifar10(root, train=True, batch_size=50, dataidxs=None):
    train_transform, valid_transform = _data_transforms_cifar10()

    if train:
        dataset = CIFAR10_truncated(
            root, dataidxs, train=True, transform=train_transform, download=False
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
        )
        dataset.targets

    else:
        dataset = CIFAR10_truncated(
            root, dataidxs, train=False, transform=train_transform, download=False
        )

#         dataset = CIFAR10_truncated(
#             root, dataidxs, train=False, transform=valid_transform, download=False
#         )

        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=5
        )

    return dataloader


# def get_dataloader_cifar5(root, train=True, batch_size=50, dataidxs=None):
#     with HiddenPrints():
#         train_transform, valid_transform = _data_transforms_cifar10()
#         cifar10_targets = get_all_targets_cifar10(root, train)
#         original_indices = (cifar10_targets < 5).nonzero()[0]

#         if dataidxs is not None:
#             dataidxs = original_indices[dataidxs]
#         else:
#             dataidxs = original_indices

#         if train:
#             dataset = CIFAR10_truncated(
#                 root, dataidxs, train=True, transform=train_transform, download=False
#             )
#             dataloader = data.DataLoader(
#                 dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
#             )
#             dataset.targets

#         else:
#             dataset = CIFAR10_truncated(
#                 root, dataidxs, train=False, transform=train_transform, download=False
#             )
#             dataloader = data.DataLoader(
#                 dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=5
#             )

#         return dataloader


# class HiddenPrints:
#     def __enter__(self):
#         self._original_stdout = sys.stdout
#         sys.stdout = open(os.devnull, "w")

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._original_stdout
