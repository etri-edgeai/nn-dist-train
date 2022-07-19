import torch
import torch.nn as nn

__all__ = ['cross_entropy']


def cross_entropy(pred, target, device):
    criterion = nn.CrossEntropyLoss().to(device)
            
    return criterion(pred, target)
