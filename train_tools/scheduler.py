import math
import numpy as np

__all__ = ['multistep_lr_scheduler']


def multistep_lr_scheduler(args, optimizer, epoch):
    lr = args.local_lr
    steps = np.sum(epoch > np.asarray(args.milestones))
    if steps > 0:
        lr = lr * (args.lr_decay ** steps)
            
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return optimizer