import torch

__all__ = ['weight_decay', 'fedprox']


def weight_decay(model, lamb):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.add_(p.data, alpha=lamb)
            
    return model


def fedprox(model, mu, server_weight):
    for name, p in model.named_parameters():
        p.grad.add_(torch.abs(p.data - server_weight[name].data).mul(mu))
        
    return model
