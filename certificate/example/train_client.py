# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

torch.set_printoptions(10)

from server import *
from local import *
from train_tools import *
from train_tools.criterion import OverhaulLoss
from utils import *
from utils import objectview
from train_tools.models.layer3 import Layer3

import os
import copy
import numpy as np
import argparse, json, os, random
import warnings
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


    
##############################################################################################
SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}

def _get_setups(opt):
    # Fix randomness
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    
    # train setups
    model = Layer3()

    criterion = OverhaulLoss(**opt.criterion.params.__dict__)
    optimizer = optim.SGD(model.parameters(), **opt.optimizer.params.__dict__)
    scheduler = None

    if opt.scheduler.enabled:
        scheduler = SCHEDULER[opt.scheduler.type](
            optimizer, **opt.scheduler.params.__dict__
        )

    return model, criterion, optimizer, scheduler


################################################################################################
def main():
    
    # Fix randomness
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    # Setups
    model, criterion, optimizer, scheduler = _get_setups(opt)
    
    trainset = torch.load('./data/trainset.pth')
    dataloader = DataLoader(trainset, batch_size=128, shuffle=True)

    try:
        client_weight = torch.load('./checkpoint/client_ckpt.pth')
        print("[train_client.py] load trained checkpoint")
    except:
        print("[train_client.py] there is no trained checkpoint")
        client_weight = None

    try:
        optimizer_weight = torch.load('./optimizer/optimizer.pth', map_location=opt.device)  
    except:
        optimizer_weight = None

    local_args = opt.fed_setups.local_params
    if local_args is not None:
        local = Local(
            model,
            criterion,
            dataloader,
            optimizer,
            local_args.local_ep,
            local_args.local_bs,
            global_loss=[local_args.global_loss, local_args.global_alpha]
            )
    else:
        local = Local(
            model,
            criterion,
            dataloader,
            optimizer,
        )

    if client_weight is not None:
        local.model.load_state_dict(client_weight)
    if optimizer_weight is not None:
        local.optimizer.load_state_dict(optimizer_weight)

    local_results = local.train()
    print(local_results)

    trained_optimizer_weight = copy.deepcopy(local.optimizer.state_dict())
    trained_client_weight = copy.deepcopy(local.model.state_dict())
    
    torch.save(trained_optimizer_weight, './optimizer/optimizer.pth')
    torch.save(trained_client_weight, './checkpoint/client_ckpt.pth')


if __name__ == "__main__":

    if not os.path.isdir('./checkpoint'):
        os.makedirs('./checkpoint')
    if not os.path.isdir('./optimizer'):
        os.makedirs('./optimizer')

    # Parser arguments for terminal execution
    parser = argparse.ArgumentParser(description="Process Config Dicts")
    parser.add_argument("--config_path", default="./config/default.py", type=str)
    args = parser.parse_args()

    # Load a configuration file
    with open(args.config_path) as f:
        config_code = f.read()
        exec(config_code)
    
    opt = objectview(configdict)

    main()
