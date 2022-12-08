import torch
import torch.nn as nn
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from train_tools.measures import *

__all__ = ["ClientTrainer"]


class ClientTrainer:
    def __init__(self, model, local_epochs, device, num_classes):
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        # model & optimizer
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0) #임시로 지정한 값, global에 의해서 얼마든 바뀔 수 있음!!
        self.criterion = nn.CrossEntropyLoss()
        self.local_epochs = local_epochs
        self.device = device
        self.datasize = None # client index에 맞게 채워질 예정
        self.num_classes = num_classes
        self.trainloader = None # client index에 맞게 채워질 예정
        self.testloader = None # client index에 맞게 채워질 예정

    def train(self):
        """Local training"""

        self.model.train()
        self.model.to(self.device)

        local_results = {}
        local_size = self.datasize

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, targets)

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
            _
        ) = evaluate_model_classwise(
            self.model,
            self.testloader,
            self.num_classes,
            get_logits=False,
            device=self.device,
        )
        
        
#         local_results["train_acc"] = evaluate_model(
#             self.model, self.trainloader, self.device
#         )
#         (
#             local_results["classwise_accuracy"],
#             local_results["test_acc"],
#             local_results["logits_vec"],
#         ) = evaluate_model_classwise(
#             self.model,
#             self.testloader,
#             self.num_classes,
#             get_logits=True,
#             device=self.device,
#         )

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
