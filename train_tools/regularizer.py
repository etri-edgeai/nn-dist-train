import torch
import copy

__all__ = ['weight_decay']


def weight_decay(model, lamb):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.add_(p.data, alpha=lamb)
            
    return model
