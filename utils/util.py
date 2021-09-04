import random
import torch
import numpy as np

__all__ = ['fix_seed', 'set_path']


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def set_path(args):
    # make dirs
    for tmp_path in ['./logs', './model_checkpoints']:
        path = os.path.join(tmp_path, args.dataset, args.model, args.dist_mode)
        if not os.path.isdir(path):
            os.makedirs(path)
    
    # set file path
    args.log_path = './logs/%s/%s/non_iid_%s/[%s]seed%s_clients%s-%s_rounds%s_epochs%s' % (args.dataset, args.model, args.non_iid, args.algorithm, args.seed, args.num_clients, args.clients_per_round, args.num_epochs)
    if args.exp_name:
        args.log_path += '_%s' % args.exp_name
    args.log_path += '.csv'
    
    args.checkpoint_path = './model_checkpoints/%s/%s/non_iid_%s/[%s]seed%s_clients%s-%s_rounds%s_epochs%s' % (args.dataset, args.model, args.non_iid, args.algorithm, args.seed, args.num_clients, args.clients_per_round, args.num_epochs)
    if args.exp_name:
        args.checkpoint_path += '_%s' % args.exp_name
    args.checkpoint_path += '.pth'
    
    args.img_path = './img/%s/%s/non_iid_%s/[%s]seed%s_clients%s-%s_rounds%s_epochs%s' % (args.dataset, args.model, args.non_iid, args.algorithm, args.seed, args.num_clients, args.clients_per_round, args.num_epochs)
    if args.exp_name:
        args.img_path += '_%s' % args.exp_name
    if not os.path.isdir(args.img_path):
        os.mkdirs(args.img_path)
        
    log_columns = ['test_acc', 'est_std', 'class_min_acc', 'class_max_acc']
    log_pd = pd.DataFrame(np.zeros([args.num_rounds + 1, len(log_columns)]), columns = log_columns)

    return args, log_pd
    