import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

from train_tools.preprocessing.cifar10.loader import get_dataloader_cifar10
from train_tools.preprocessing.cifar100.loader import get_dataloader_cifar100
from train_tools.preprocessing.tinyimagenet.loader import get_dataloader_tinyimagenet
from train_tools.preprocessing.imagenet.loader import get_dataloader_imagenet

__all__ = ["ClientTrainer"]

DATA_LOADERS = {
    "cifar10": get_dataloader_cifar10,
    "cifar100": get_dataloader_cifar100,
    "tinyimagenet": get_dataloader_tinyimagenet,
    "imagenet": get_dataloader_imagenet
    
}

class ClientTrainer(BaseClientTrainer):
    def __init__(self, moon_criterion, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        self.moon_criterion = moon_criterion

    def train(self):
        """Local training"""

        # Keep global model and prev local model
        self._keep_global()
        self._keep_prev_local()

        self.model.train()
        self.model.to(self.device)
        
        local_results = {}
        local_size = self.datasize
        root = os.path.join("./data", self.data_name)
        self.trainloader=DATA_LOADERS[self.data_name](root=root, train=True, batch_size=50, dataidxs=self.train_idxs)
        if self.test_idxs is None: #LDA Setting
            for _ in range(self.local_epochs*self.local_epochs):
                dataiter = iter(self.trainloader)
                data, targets = next(dataiter)
                
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)


                z=self.model.extract_features(data)
                # for moon contrast
                z_prev = self.prev_model.extract_features(data)
                z_g = self.dg_model.extract_features(data)

                loss = self.moon_criterion(output, targets, z, z_prev, z_g)

                # backward pass
                loss.backward()
                self.optimizer.step()
            
        else:
            for _ in range(self.local_epochs):
                for data, targets in self.trainloader:
                    self.optimizer.zero_grad()

                    # forward pass
                    data, targets = data.to(self.device), targets.to(self.device)
                    output = self.model(data)


                    z=self.model.extract_features(data)
                    # for moon contrast
                    z_prev = self.prev_model.extract_features(data)
                    z_g = self.dg_model.extract_features(data)

                    loss = self.moon_criterion(output, targets, z, z_prev, z_g)

                    # backward pass
                    loss.backward()
                    self.optimizer.step()
        local_results = self._get_local_stats()            
                

        return local_results, local_size

    def download_global(self, server_weights, server_optimizer, prev_weights):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        server_optimizer_info=server_optimizer['param_groups'][0]
    
        body_params = [p for name, p in self.model.named_parameters() if 'classifier' not in name]
        head_params = [p for name, p in self.model.named_parameters() if 'classifier' in name]
    
        self.optimizer= torch.optim.SGD([{'params': body_params, 'lr': server_optimizer_info['lr']},
                                     {'params': head_params, 'lr': server_optimizer_info['lr']}],
                                    momentum=server_optimizer_info['momentum'],
                                    weight_decay=server_optimizer_info['weight_decay'])
        self.prev_weights = prev_weights

    def _keep_prev_local(self):
        """Keep distributed global model's weight"""
        self.prev_model = copy.deepcopy(self.model)
        self.prev_model.load_state_dict(self.prev_weights)
        self.prev_model.to(self.device)

        for params in self.prev_model.parameters():
            params.requires_grad = False