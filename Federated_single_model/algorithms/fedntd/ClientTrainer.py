import torch
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from train_tools.measures import *

__all__ = ["ClientTrainer"]


class ClientTrainer:
    def __init__(self, model, local_epochs, device, criterion, num_classes):
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        # model & optimizer
        self.model = model
        self.dg_model = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.criterion = criterion
        self.local_epochs = local_epochs
        self.device = device
        self.datasize = None
        self.num_classes = num_classes
        self.trainloader = None
        self.testloader = None

    def train(self):
        """Local training"""

        # Keep global model's weights
        self._keep_global()

        self.model.train()
        self.model.to(self.device)

        local_results = {}
        local_size = self.datasize

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                logits, dg_logits = self.model(data), self._get_dg_logits(data)
                loss = self.criterion(logits, targets, dg_logits)

                # backward pass
                loss.backward()
                self.optimizer.step()

        # Local train accuracy
        local_results["train_acc"] = evaluate_model(
            self.model, self.trainloader, self.device
        )
        (
            local_results["classwise_accuracy"],
            local_results["test_acc"],
            local_results["logits_vec"],
        ) = evaluate_model_classwise(
            self.model,
            self.testloader,
            self.num_classes,
            get_logits=True,
            device=self.device,
        )

        return local_results, local_size

    def download_global(self, server_weights, server_optimizer):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)

    def upload_local(self):
        """Uploads local model's parameters"""
        local_weights = copy.deepcopy(self.model.state_dict())

        return local_weights

    def reset(self):
        """Clean existing setups"""
        self.datasize = None
        self.trainloader = None
        self.testloader = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)

    def _keep_global(self):
        """Keep distributed global model's weight"""
        self.dg_model = copy.deepcopy(self.model)
        self.dg_model.to(self.device)

        for params in self.dg_model.parameters():
            params.requires_grad = False

    def _get_dg_logits(self, data):
        with torch.no_grad():
            dg_logits = self.dg_model(data)

        return dg_logits
