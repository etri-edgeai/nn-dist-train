import copy
import torch
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
            for ind, (inputs, labels) in enumerate(client_loader['train'][client]):
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
