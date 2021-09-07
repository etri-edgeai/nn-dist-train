import copy
import torch
import itertools
from utils.util import gpu_to_cpu, cpu_to_gpu
from .criterion import cross_entropy

__all__ = ['client_opt']


CRITERION = {'ce': cross_entropy}
OPTIMIZER = {}
CLSCHEDULER = {}


# train local clients
def client_opt(args, client_loader, client_datasize, model, weight, momentum, rounds):
    # argument for training clients
    server_weight, client_weight = weight['server'], weight['client']
    client_momentum = momentum['client']
    
    criterion = CRITERION[args.local_criterion]
    optimizer = OPTIMIZER[args.local_optimizer]
    
    lr = scheduler(args.local_lr, rounds, args.lr_decay, [int(epo) for epo in args.milestones.split(',')], args.sch_type)
    num_clients = CLSCHEDULER[args.cl_scheduler](round(float(eval(args.clients_per_round))), rounds, args)
#     num_clients = uniform(round(float(eval(args.clients_per_round))), rounds, args)

    # regularization
    momentum = args.local_momentum
    # only use nesterov True update
    nesterov = True
    wd = args.wd
    mu = args.mu
    
    clients = client_loader['train'].keys()
    selected_clients = client_selection(clients, num_clients, args, client_datasize)
    print('[%s algorithm] %s clients are selected' % (args.algorithm, selected_clients))
    
#     prev_momentum = copy.deepcopy(client_momentum)
    for client in set(selected_clients):
        # load client weights
        client_weight[client] = cpu_to_gpu(client_weight[client], args.device)
        model.load_state_dict(client_weight[client])
        if args.logit_distillation:
            tmp_model = copy.deepcopy(model)
            tmp_model.eval()
            for param in tmp_model.parameters():
                param.requires_grad = False
        
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
                
                with torch.set_grad_enabled(True):
                    # Forward pass
                    pred  = model(inputs)
                    loss = criterion(pred, labels, args.device)

                    # Backward pass (compute the gradient graph)
                    loss.backward()
                    
                    # regularization term
                    if wd:
                        model = weight_decay(model, wd)
                        
                    if mu and not sub_model:
                        server_weight = cpu_to_gpu(server_weight, args.device)
                        model = fedprox(model, mu, server_weight)
                        server_weight = gpu_to_cpu(server_weight)
                    
                    if momentum:
                        model, client_momentum = apply_local_momentum(model, momentum, nesterov, client_momentum, client, args.device)
                        
                    model = optimizer(model, lr)
                    
        # after local training
        client_weight[client] = gpu_to_cpu(copy.deepcopy(model.state_dict()))

    return client_weight, selected_clients, client_momentum
