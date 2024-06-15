import copy
import os
import sys
import torch
import math
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.feddr.ClientTrainer import ClientTrainer
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
        
        self.model=copy.deepcopy(self.etf(self.model))
        print("\n>>> FedDR Server initialized...\n")

    def etf(self, model):
        for name, param in model.named_parameters():
            
            if 'classifier' in name:
                class_num=param.shape[0]
                feature_dim=param.shape[1]


                identity_matrix=torch.eye(class_num)
                one_vector=torch.ones(class_num)

                A=identity_matrix-(1/class_num)* (one_vector.view(-1, 1) @ one_vector.view(1, -1))

                B, _ = np.linalg.qr(copy.deepcopy(param.data.T.cpu()).numpy())
                B = torch.tensor(B).float() #(d,c)
                assert torch.allclose(torch.matmul(B.T, B), torch.eye(class_num), atol=1e-06), torch.max(torch.abs(torch.matmul(B.T, B) - torch.eye(class_num)))

                C=math.sqrt(class_num/(class_num-1))

                etf=C*(B @ A)
                param.data=copy.deepcopy(etf.T.detach())

        return model
