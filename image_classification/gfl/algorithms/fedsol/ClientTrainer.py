import torch
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

from train_tools.preprocessing.cifar10.loader import get_dataloader_cifar10
from train_tools.preprocessing.cifar100.loader import get_dataloader_cifar100

__all__ = ["ClientTrainer"]

DATA_LOADERS = {
    "cifar10": get_dataloader_cifar10,
    "cifar100": get_dataloader_cifar100,
}



class ClientTrainer(BaseClientTrainer):
    def __init__(self, criterion, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        self.criterion = criterion

    def train(self):
        """Local training"""

        # Keep global model's weights
        self._keep_global()

        self.model.train()
        self.model.to(self.device)
        
        local_results = {}
        local_size = self.datasize
        root = os.path.join("./data", self.data_name)
        self.trainloader=DATA_LOADERS[self.data_name](root=root, train=True, batch_size=50, dataidxs=self.train_idxs)
        
        if self.test_idxs is None: #LDA Setting
            for _ in range(self.local_epochs):
                for data, targets in self.trainloader:
                    data, targets = data.to(self.device), targets.to(self.device)

                    # first forward-backward pass
                    enable_running_stats(self.model)

                    if not self.perturb_head:
                        freeze_head(self.model)

                    if not self.perturb_body:
                        freeze_body(self.model)

                    data, targets = data.to(self.device), targets.to(self.device)
                    logits, dg_logits = self.model(data), self._get_dg_logits(data)

                    with torch.no_grad():
                        dg_probs = torch.softmax(dg_logits / 3, dim=1)
                    pred_probs = F.log_softmax(logits / 3, dim=1)

                    loss = self.KLDiv(pred_probs, dg_probs)
                    loss.backward()

                    if not self.perturb_head:
                        zerograd_head(self.model)

                    if not self.perturb_body:
                        zerograd_body(self.model)

                    self.sam_optimizer.first_step(zero_grad=True)

                    unfreeze(self.model)

                    # second forward-backward pass
                    disable_running_stats(self.model)
                    self.criterion(
                        self.model(data), targets
                    ).backward()  # make sure to do a full forward pass
                    self.sam_optimizer.second_step(zero_grad=True)

            local_results = self._get_local_stats()
        
        else:
            for _ in range(self.local_epochs):
                for data, targets in self.trainloader:
                    data, targets = data.to(self.device), targets.to(self.device)

                    # first forward-backward pass
                    enable_running_stats(self.model)

                    if not self.perturb_head:
                        freeze_head(self.model)

                    if not self.perturb_body:
                        freeze_body(self.model)

                    data, targets = data.to(self.device), targets.to(self.device)
                    logits, dg_logits = self.model(data), self._get_dg_logits(data)

                    with torch.no_grad():
                        dg_probs = torch.softmax(dg_logits / 3, dim=1)
                    pred_probs = F.log_softmax(logits / 3, dim=1)

                    loss = self.KLDiv(pred_probs, dg_probs)
                    loss.backward()

                    if not self.perturb_head:
                        zerograd_head(self.model)

                    if not self.perturb_body:
                        zerograd_body(self.model)

                    self.sam_optimizer.first_step(zero_grad=True)

                    unfreeze(self.model)

                    # second forward-backward pass
                    disable_running_stats(self.model)
                    self.criterion(
                        self.model(data), targets
                    ).backward()  # make sure to do a full forward pass
                    self.sam_optimizer.second_step(zero_grad=True)

            local_results = self._get_local_stats()    
            

        return local_results, local_size

    def download_global(self, server_weights, server_optimizer):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)
        self.sam_optimizer = self._get_sam_optimizer(self.optimizer)

    def _get_sam_optimizer(self, base_optimizer):
        optim_params = base_optimizer.state_dict()
        lr = optim_params["param_groups"][0]["lr"]
        momentum = optim_params["param_groups"][0]["momentum"]
        weight_decay = optim_params["param_groups"][0]["weight_decay"]
        sam_optimizer = SAM(
            self.model.parameters(),
            base_optimizer=torch.optim.SGD,
            rho=self.rho,
            adaptive=False,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        return sam_optimizer

    @torch.no_grad()
    def _get_dg_logits(self, data):
        dg_logits = self.dg_model(data)

        return dg_logits

