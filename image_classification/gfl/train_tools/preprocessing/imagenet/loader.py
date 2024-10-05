import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def _data_transforms_imagenet():



    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224), 
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),             
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


    valid_transform = transforms.Compose([
    transforms.Resize(256),              
    transforms.CenterCrop(224),          
    transforms.ToTensor(),               
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

    return train_transform, valid_transform



def get_all_targets_imagenet(root, train=True):
    
    train_transform, valid_transform = _data_transforms_imagenet()

    if train:
        dataset = datasets.ImageFolder(root='data/ImageNet100/train', transform=train_transform)
        all_targets = np.array(dataset.targets)
    if not train:
        dataset = datasets.ImageFolder(root='data/ImageNet100/val', transform=valid_transform)
        all_targets = np.array(dataset.targets)
        
    return all_targets





def get_dataloader_imagenet(root, train=True, batch_size=50, dataidxs=None):
    train_transform, valid_transform = _data_transforms_imagenet()
    if train:
        dataset = datasets.ImageFolder(root='data/ImageNet100/train', transform=train_transform)
        if dataidxs is not None:
            dataloader = DataLoader(DatasetSplit(dataset, dataidxs), batch_size=batch_size, shuffle=True)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            

    else:
        dataset = datasets.ImageFolder(root='data/ImageNet100/val', transform=valid_transform)
        if dataidxs is not None:
            dataloader = DataLoader(DatasetSplit(dataset, dataidxs), batch_size=batch_size, shuffle=False)
        else:
            dataloader = DataLoader(dataset, batch_size=50, shuffle=False)
            
    return dataloader


