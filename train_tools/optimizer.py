import copy
import torch

__all__ = ['sgd']

    
def sgd(model, lr):
    for name, p in model.named_parameters():
        if p.grad is not None:
            p.data.sub_(p.grad, alpha=lr)

    return model