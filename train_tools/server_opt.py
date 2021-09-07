import copy
import torch
from utils.util import gpu_to_cpu, cpu_to_gpu
from .optimizer import pseudo_sgd, apply_global_momentum

__all__ = ['server_opt']


def server_opt(args, client_loader, client_datasize, model, weight, momentum, selected_clients):
    server_weight, client_weight = weight['server'], weight['client']
    server_momentum = momentum['server']
        
    # initialize the trained results
    delta_weight = {}
    for client in client_loader['train']:
        delta_weight[client] = {}
    
    # get delta_weight of each client
    for client in client_loader['train']:
        if client in selected_clients:
            for name, p in client_weight[client].items():
                delta_weight[client][name] = (server_weight[name].data - p.data).cpu()
        else:
            for name in client_weight[client].keys():
                delta_weight[client][name] = torch.zeros_like(server_weight[name]).cpu()
                
    # get aggregate weights
    if args.full_part:
        total_datasize = sum(client_datasize)
            
        aggregate_weights = {}
        for client in client_loader['train']:
            aggregate_weights[client] = client_datasize[client] / total_datasize
    
    else:
        if args.algorithm == 'fedprox':
            aggregate_weights = {}
            for client in selected_clients:
                if aggregate_weights.get(client):
                    aggregate_weights[client] += 1 / len(selected_clients)
                else:
                    aggregate_weights[client] = 1 / len(selected_clients)

        else:
            # fedavg_pdp algorithm
            total_datasize = 0
            for client in selected_clients:
                total_datasize += client_datasize[client]

            aggregate_weights = {}
            for client in selected_clients:
                aggregate_weights[client] = client_datasize[client] / total_datasize    
    # get delta weights
    delta_dict = {}
    for client, agg in aggregate_weights.items():
        for name, delta in delta_weight[client].items():
            if delta_dict.get(name) is None:
                delta_dict[name] = (delta.data * agg).to(args.device)
            else:
                delta_dict[name] += (delta.data * agg).to(args.device)
                
    # server momentum
    if args.global_momentum:
        delta_dict, server_momentum = apply_global_momentum(args, delta_dict, server_momentum)
    
    # update server weights
    server_weight = cpu_to_gpu(server_weight, args.device)
    model.load_state_dict(server_weight)
    model = pseudo_sgd(model, delta_dict)
    
    # update server weight and clients weight
    server_weight = gpu_to_cpu(copy.deepcopy(model.state_dict()))
    for client in client_weight.keys():
        client_weight[client] = copy.deepcopy(server_weight)

    weight['server'] = server_weight
    weight['client'] = client_weight
    momentum['server'] = server_momentum
    
    return weight, momentum, model