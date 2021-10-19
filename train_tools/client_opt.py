import copy
import numpy as np
import torch
import torch.nn.functional as F
from utils.util import gpu_to_cpu, cpu_to_gpu
from .criterion import cross_entropy
from .optimizer import sgd, apply_local_momentum
from .regularizer import weight_decay, fedprox
from .scheduler import multistep_lr_scheduler, cosine_lr_scheduler
from .client_selection import client_selection

__all__ = ['client_opt']


CRITERION = {'ce': cross_entropy}
OPTIMIZER = {'sgd': sgd}
SCHEDULER = {'multistep': multistep_lr_scheduler, 'cosine': cosine_lr_scheduler}


def dataset_update(train_loader, model, args):
    ret,idx_list,label_list =[],[],[]
    model.eval()
    train_loader.dataset.sample_deterministic()

    with torch.no_grad():
        for idx, [x, y, i] in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)

            logit = model(x)
            prob = F.softmax(logit, dim=1)

            ret.extend(torch.gather(prob,1, y.unsqueeze(1)).detach().cpu().reshape(-1))
            idx_list.extend(i.squeeze())
            label_list.extend(y.squeeze().detach().cpu().reshape(-1))
            
    train_loader.dataset.sample_probabilistic()

    
    # label
    idx_list = torch.tensor(idx_list).reshape(-1)
    label_list = torch.tensor(label_list).reshape(-1)
    
    sort_pos = idx_list.argsort(descending=False)
    label_list = label_list[sort_pos]
    
    weight = torch.zeros_like(label_list)
    for cidx in range(torch.max(label_list)+1):
        pos = torch.where(label_list == cidx)[0]
        weight[pos] = len(weight) / len(pos) if len(pos) != 0 else 0
    
    print(weight.shape, weight)
    weight = weight.float()
    weight /= torch.sum(weight)
    
    weight = torch.ones_like(weight).float()/len(weight)
        
    # Softmax
#     weight = torch.tensor(ret)
#     weight = 1. / (weight + 0.01)
#     weight /= torch.sum(weight)
    
    
    
    train_loader.dataset.update_prob(weight.detach())
    print(weight, torch.max(weight), torch.mean(weight), torch.std(weight))
    return train_loader

    
# train local clients
def client_opt(args, client_loader, client_datasize, model, weight, momentum, rounds):
    # argument for training clients
    server_weight, client_weight = weight['server'], weight['client']
    client_momentum = momentum['client']
    
    criterion = CRITERION[args.local_criterion]
    optimizer = OPTIMIZER[args.local_optimizer]
    lr = SCHEDULER[args.scheduler](args, rounds)
    
    selected_clients = client_selection(args, client_datasize)
    print('[%s algorithm] %s clients are selected' % (args.algorithm, selected_clients))
    
    for client in set(selected_clients):
        # load client weights
        client_weight[client] = cpu_to_gpu(client_weight[client], args.device)
        model.load_state_dict(client_weight[client])
        
        # local training
        model.train()
        for epoch in range(args.num_epochs):
            if args.self_balancing and rounds >= args.fedcsb_warmup:
                client_loader['train'][client] = dataset_update(client_loader['train'][client], model, args)
            
            for ind, data in enumerate(client_loader['train'][client]):
                if not args.self_balancing:
                    inputs, labels = data
                else:
                    inputs, labels, _ = data
                print(labels)
                # more develpment required
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                # optimizer.zero_grad()
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
                
                # Forward pass
                pred  = model(inputs)
                loss = criterion(pred, labels, args.device)

                # Backward pass (compute the gradient graph)
                loss.backward()

                # regularization term
                if args.wd:
                    model = weight_decay(model, args.wd)
                # fedprox algorithm
                if args.mu:
                    server_weight = cpu_to_gpu(server_weight, args.device)
                    model = fedprox(model, args.mu, server_weight)
                    server_weight = gpu_to_cpu(server_weight)
                # sgd with momentum
                if args.local_momentum:
                    model, client_momentum = apply_local_momentum(args, model, client, client_momentum)

                model = optimizer(model, lr)
                    
        # after local training
        client_weight[client] = gpu_to_cpu(copy.deepcopy(model.state_dict()))
    
    weight['client'] = client_weight
    momentum['client'] = client_momentum
    
    return weight, momentum, selected_clients
