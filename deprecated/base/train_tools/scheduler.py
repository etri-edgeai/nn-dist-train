import math
import numpy as np

__all__ = ['multistep_lr_scheduler', 'cosine_lr_scheduler']


def multistep_lr_scheduler(args, epoch):
    lr = args.local_lr
    steps = np.sum(epoch > np.asarray(args.milestones))
    if steps > 0:
        lr = lr * (args.lr_decay ** steps)
        
    return lr


def cosine_lr_scheduler(args, epoch):
    lr = args.local_lr
    eta_min = lr * (args.lr_decay ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / args.num_epochs)) / 2
            
    return lr