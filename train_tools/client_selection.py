import random
import numpy as np
from numpy.random import choice

__all__ = ['client_selection']


def client_selection(args, client_datasize=None):
    clients = range(client_datasize.size)
    
    if args.algorithm == 'centralized':
        selected_clients = clients
    elif args.algorithm in ['fedavg', 'fedavg_pdp']:
        selected_clients = random.sample(clients, args.clients_per_round)
    elif args.algorithm == 'fedprox':
        weight_dist = [num / sum(client_datasize) for num in client_datasize]
        selected_clients = choice(range(len(clients)), args.clients_per_round, p=weight_dist)
    else:
        raise NotImplemented
        
    return selected_clients