import copy
import os
import sys
import torch
import math
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.spherefed.ClientTrainer import ClientTrainer
from algorithms.BaseServer import BaseServer

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        """
        Server class controls the overall experiment.
        """
        self.client = ClientTrainer(
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )
        
        self.model=copy.deepcopy(self.gram_schmidt(self.model))
        print("\n>>> SphereFed Server initialized...\n")

    def gram_schmidt(self, model):
        for name, param in model.named_parameters():
            if 'classifier' in name:
                u=torch.zeros(param.shape)
                for i in range(param.shape[0]):
                    if i==0:
                        u[i]=param[0]/torch.norm(param[0])
                    else:
                        v=copy.deepcopy(param[i].detach())
                        for j in range(i):
                            v-=(torch.inner(param[i],u[j])/torch.norm(u[j]))*u[j]
                        u[i]=v/torch.norm(v)
#                 print(torch.matmul(u, torch.transpose(u, 0, 1)))
#                 print(torch.diagonal(torch.matmul(u, torch.transpose(u, 0, 1)), 0))
#                 print(param.data)        
                param.data=copy.deepcopy(u.detach())

        return model
