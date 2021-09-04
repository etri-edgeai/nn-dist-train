import random
import torch
import numpy as np

__all__ = ['fix_seed']


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)