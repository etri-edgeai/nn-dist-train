import torch
import os
import sys
from torch import nn
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
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        
        self.loss_func = nn.MSELoss()
    
    def download_global(self, server_weights, server_optimizer):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)

        server_optimizer_info=server_optimizer['param_groups'][0]
    
        body_params = [p for name, p in self.model.named_parameters() if 'classifier' not in name]
        head_params = [p for name, p in self.model.named_parameters() if 'classifier' in name]
    
        self.optimizer= torch.optim.SGD([{'params': body_params, 'lr': server_optimizer_info['lr']},
                                     {'params': head_params, 'lr': 0}],
                                    momentum=server_optimizer_info['momentum'],
                                    weight_decay=server_optimizer_info['weight_decay'])
    
        self.MSE = nn.MSELoss()
    
    
    def train(self):
        """Local training"""
        
        # Keep global model's weights
        self.teacher=copy.deepcopy(self.model)
        self.teacher.to(self.device)
        self.teacher.eval()
        
        for params in self.teacher.parameters():
            params.requires_grad = False
        
        self.model.train()
        self.model.to(self.device)
        
        local_size = self.datasize
        
        epoch_loss = []
        
        root = os.path.join("./data", self.data_name)
        self.trainloader=DATA_LOADERS[self.data_name](root=root, train=True, batch_size=self.batch_size, dataidxs=self.train_idxs)
        
        etf_label=self.model.state_dict()['classifier.weight']
        if self.test_idxs is None: #LDA Setting
            for _ in range(self.average_iteration*self.local_epochs):
                dataiter = iter(self.trainloader)
                data, targets = next(dataiter)
                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                
                etf_labels=etf_label[targets].to(self.device) 
                
                dg_original_features= self.teacher.extract_original_features(data)

                self.model.zero_grad()

                
                features = self.model.extract_features(data)
                original_features=self.model.extract_original_features(data)
                
                
                inner_products=torch.sum(features*etf_labels, dim=1)
                loss = torch.mean((0.9)*(1/2)*self.MSE(inner_products,torch.ones(len(targets)).to(self.device))+(0.1)*self.MSE(original_features, dg_original_features))
                
                # backward pass
                loss.backward()

                self.optimizer.step()
                
        else: #Sharding Setting
            for _ in range(self.local_epochs):
                batch_loss=[]
                for data, targets in self.trainloader:

                    # forward pass
                    data, targets = data.to(self.device), targets.to(self.device)
                    etf_labels=etf_label[targets].to(self.device) 
                    dg_original_features= self.teacher.extract_original_features(data)
                    
                    
                    self.model.zero_grad()
                    
                    features = self.model.extract_features(data)
                    original_features=self.model.extract_original_features(data)


                    inner_products=torch.sum(features*etf_labels, dim=1)

                    loss = torch.mean((0.9)*(1/2)*self.MSE(inner_products,torch.ones(len(targets)).to(self.device))+(0.1)*self.MSE(original_features, dg_original_features))

                    # backward pass
                    loss.backward()

                    self.optimizer.step()
                    
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        local_results = self._get_local_stats()
        
    
        return local_results, local_size
    
    
    
    
    