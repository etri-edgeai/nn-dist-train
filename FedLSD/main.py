# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

torch.set_printoptions(10)

from server import *
from train_tools import *
from utils import *

import numpy as np
import wandb, argparse, json, os, random
import warnings

warnings.filterwarnings("ignore")

# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Process Config Dicts")
parser.add_argument("--config_path", default="./config/default.py", type=str)
args = parser.parse_args()

# Load a configuration file
with open(args.config_path) as f:
    config_code = f.read()
    exec(config_code)

def _get_setups(opt):
    # datasets
    datasetter = DataSetter(**opt.data_setups.dataset_params.__dict__)
    datasets = datasetter.data_distributer(
        **opt.data_setups.distribute_params.__dict__,
        n_clients=opt.fed_setups.server_params.n_clients,
    pass


################################################################################################
def main():
    # Setups
    datasets, model, criterion, optimizer, evaluator, scheduler = _get_setups(opt)
     
    pass


if __name__ == "__main__":
    opt = objectview(configdict)

    # Initialize wandb
    wandb.init(
        project=opt.exp_info.project_name,
        name=opt.exp_info.name,
        tags=opt.exp_info.tags,
        group=opt.exp_info.group,
        notes=opt.exp_info.notes,
        config=configdict,
    )

    main()
