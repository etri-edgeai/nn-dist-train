import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import algorithms
from train_tools import *
from utils import *

import numpy as np
import argparse
import warnings
import wandb
import random
import pprint
import os
import sys

warnings.filterwarnings("ignore")

# Set torch base print precision
torch.set_printoptions(10)

ALGO = {
    "fedavg": algorithms.fedavg.Server,
    "fedavgm": algorithms.fedavgm.Server,
    "fedprox": algorithms.fedprox.Server,
    "fedntd": algorithms.fedntd.Server,
    "feddual": algorithms.feddual.Server,
    "fedcont": algorithms.fedcont.Server,
    "scaffold": algorithms.scaffold.Server,
    "moon": algorithms.moon.Server,
    "fednova": algorithms.fednova.Server,
    "feddyn": algorithms.feddyn.Server,
    "fedcurv": algorithms.fedcurv.Server,
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}


def _get_setups(args):
    """Get train configuration"""
    # Fix randomness for data distribution
    np.random.seed(19940817)
    random.seed(19940817)

    data_distributed = data_distributer(**args.data_setups) #datasetter.py에 있는 fct, Server Class의 attribute로 들어간다.
#  {"global": global_loaders,"local": local_loaders,"data_map": data_map,"num_classes": num_classes} 형태!!

    # Fix randomness for experiment
    _random_seeder(args.train_setups.seed)
    
    #create_models는 train_tools/utils.py 에 있음!!
    model = create_models(
        args.train_setups.model.name,
        args.data_setups.dataset_name,
        **args.train_setups.model.params
    )#Server Class의 attribute로 들어간다.

    optimizer = optim.SGD(model.parameters(), **args.train_setups.optimizer.params)#Server Class의 attribute로 들어간다.
    scheduler = None
    evaluator = None
    if args.train_setups.scheduler.enabled:
        scheduler = SCHEDULER[args.train_setups.scheduler.name](
            optimizer, **args.train_setups.scheduler.params
        )#{'name': 'step'}, {'gamma': 0.99, 'step_size': 1}, Server Class의 attribute로 들어간다.

    if args.train_setups.evaluator.enabled:
        eval_params = args.train_setups.evaluator.params #eval_params={'features': False,'predictions': True,'weights': False}        
        evaluator = Evaluator(
            model, data_distributed, eval_params, scenario=args.train_setups.scenario
        )#scenario={'device': 'cuda:0','local_epochs': 5,'n_rounds': 200,'sample_ratio': 0.1}, Server Class의 attribute로 들어간다.

    algo_params = args.train_setups.algo.params #{}, Server Class의 attribute로 들어간다.
    server = ALGO[args.train_setups.algo.name](
        model,
        data_distributed,
        optimizer,
        scheduler,
        evaluator,
        **args.train_setups.scenario,
        algo_params=algo_params
    )

    return server


def _random_seeder(seed):
    """Fix randomness"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    """Execute experiment"""
    server = _get_setups(args)

    wandb.watch(server.model, "parameters")
    server.run()

    model_path = os.path.join(wandb.run.dir, "model.pth")
    torch.save(server.model.state_dict(), model_path)

    # Upload to wandb
    wandb.save(model_path)


# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Process Configs")
parser.add_argument("--config_path", default="./config/fedntd.json", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--n_clients", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--partition_method", type=str)
parser.add_argument("--shard_ratio", type=int)
parser.add_argument("--partition_alpha", type=float)
parser.add_argument("--oracle_size", type=int)
parser.add_argument("--oracle_bs", type=int)
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_rounds", type=int)
parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--local_epochs", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--momentum", type=float)
parser.add_argument("--wd", type=float)
parser.add_argument("--algo_name", type=str)
parser.add_argument("--prox_mu", type=float)
parser.add_argument("--cl_loss_type", type=str)
parser.add_argument("--cl_tau", type=float)
parser.add_argument("--cl_beta", type=float)
parser.add_argument("--cl_lam", type=float)
parser.add_argument("--cl_k", type=int)
parser.add_argument("--avgm_beta", type=float)
parser.add_argument("--curv_size", type=int)
parser.add_argument("--curv_lambda", type=float)
parser.add_argument("--ewc_lambda", type=float)
parser.add_argument("--moon_mu", type=float)
parser.add_argument("--moon_tau", type=float)
parser.add_argument("--device", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--group", type=str)
parser.add_argument("--exp_name", type=str)
args = parser.parse_args()

#######################################################################################

if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt
    # Overwrite config by parsed arguments
    opt = config_overwriter(opt, args)
    # Print configuration dictionary pretty
    print("")
    print("=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(opt)
    print("=" * 120)
    
    # Initialize W&B
        
    wandb.init(config=opt, **opt.wandb_setups)

    # How many batches to wait before logging training status
    wandb.config.log_interval = 10

    main(opt)
