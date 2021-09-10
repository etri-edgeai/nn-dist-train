import os
import random
import torch
import numpy as np
import pandas as pd

__all__ = ['fix_seed', 'set_path', 'cpu_to_gpu', 'gpu_to_cpu', 'save_checkpoint']


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def set_path(args):
    # make dirs
    for tmp_path in ['./results/logs', './results/model_checkpoints']:
        path = os.path.join(tmp_path, args.dataset, args.model, 'non-iid_%s' % args.non_iid)
        if not os.path.isdir(path):
            os.makedirs(path)
    
    # set file path
    args.log_path = './results/logs/%s/%s/non-iid_%s/[%s]seed%s_clients%s-%s_rounds%s_epochs%s' % (args.dataset, args.model, args.non_iid, args.algorithm, args.seed, args.num_clients, args.clients_per_round, args.num_rounds, args.num_epochs)
    if args.exp_name:
        args.log_path += '_%s' % args.exp_name
    args.log_path += '.csv'
    
    args.checkpoint_path = './results/model_checkpoints/%s/%s/non-iid_%s/[%s]seed%s_clients%s-%s_rounds%s_epochs%s' % (args.dataset, args.model, args.non_iid, args.algorithm, args.seed, args.num_clients, args.clients_per_round, args.num_rounds, args.num_epochs)
    if args.exp_name:
        args.checkpoint_path += '_%s' % args.exp_name
    args.checkpoint_path += '.pth'
    
    args.img_path = './results/img/%s/%s/non-iid_%s/[%s]seed%s_clients%s-%s_rounds%s_epochs%s' % (args.dataset, args.model, args.non_iid, args.algorithm, args.seed, args.num_clients, args.clients_per_round, args.num_rounds, args.num_epochs)
    if args.exp_name:
        args.img_path += '_%s' % args.exp_name
    if not os.path.isdir(args.img_path):
        os.makedirs(args.img_path)
        
    log_columns = ['test_acc', 'est_std', 'class_min_acc', 'class_max_acc']
    log_pd = pd.DataFrame(np.zeros([args.num_rounds + 1, len(log_columns)]), columns = log_columns)

    return args, log_pd


def cpu_to_gpu(current_state, device):
    for k, v in current_state.items():
        current_state[k] = v.to(device)
    return current_state
    

def gpu_to_cpu(current_state):
    for k, v in current_state.items():
        current_state[k] = v.cpu()
    return current_state


def save_checkpoint(args, checkpoint):
    # save file
    state={}
    state['checkpoint'] = checkpoint
    torch.save(state, args.checkpoint_path)    
    print('Successfully saved' + args.checkpoint_path)