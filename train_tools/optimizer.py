import copy
import torch
from utils.util import gpu_to_cpu, cpu_to_gpu

__all__ = ['sgd', 'apply_local_momentum']

    
def sgd(model, lr):
    for name, p in model.named_parameters():
        if p.grad is not None:
            p.data.sub_(p.grad, alpha=lr)

    return model


def apply_local_momentum(args, model, client, client_momentum):
    if len(client_momentum[client]) == 0:
        for name, p in model.named_parameters():
            if p.grad is not None:
                buf = client_momentum[client][name] = torch.clone(p.grad).detach()
                if args.nesterov:
                    p.grad.add_(buf, alpha=args.local_momentum)
                else:
                    p.grad = buf
        client_momentum[client] = gpu_to_cpu(client_momentum[client])
    # in the course of training
    else:
        client_momentum[client] = cpu_to_gpu(client_momentum[client], args.device)
        for name, p in model.named_parameters():
            if p.grad is not None:
                buf = client_momentum[client][name]
                buf.mul_(args.local_momentum).add_(p.grad, alpha=1.0)
                if args.nesterov:
                    p.grad.add_(buf, alpha=args.local_momentum)
                else:
                    p.grad = buf
        client_momentum[client] = gpu_to_cpu(client_momentum[client])   
        
    return model, client_momentum