import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_tools import *
from train_tools.utils import model_evaluator

import copy, random
import numpy as np

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

__all__ = ["Local"]


class Local:
    def __init__(
        self,
        model,
        criterion,
        dataloader,
        optimizer,
        local_ep=1,
        local_bs=128,
        global_loss=["none", 0],
        server_location="cpu",
        device="cuda:0",
    ):

        # model & optimizer
        self.model = model
        self.criterion = criterion
        self.local_ep, self.local_bs = local_ep, local_bs
        self.global_loss_type, self.global_alpha = global_loss[0], global_loss[1]
        self.server_location = server_location
        self.device = device
        self.dataloader = dataloader
        self.optimizer = optimizer

    def train(self, test_evaluation=True):
        self.model.train()
        self.model.to(self.device)

        self._keep_global()

        train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0
        t_logits = None

        for ep in range(self.local_ep):
            print('[training] epohcs %s' % ep)    
            for itr, (data, targets) in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)

                if "D" in self.criterion.mode:  # 'D' stands for distillation
                    with torch.no_grad():
                        t_logits= self.round_global(data)

                loss = self.criterion(output, targets, t_logits)

                # additional loss to global
                if self.global_loss_type != "none" and self.global_alpha > 0:
                    loss += self.global_alpha * self._loss_to_round_global()

                # backward pass
                loss.backward()

                self.optimizer.step()

        # get stats from local trainset
        train_loss, train_acc = model_evaluator(
            self.model, self.dataloader, self.criterion, self.device
        )

        result = {
            "train_loss": train_loss,
            "train_acc": train_acc
        }

        self.model.to(self.server_location)

        return result

    def _keep_global(self):
        self.round_global = copy.deepcopy(self.model)
        for params in self.round_global.parameters():
            params.requires_grad = False

    def _loss_to_round_global(self):
        vec = []
        if self.global_loss_type == "proximal":
            for i, ((name1, param1), (name2, param2)) in enumerate(
                zip(self.model.named_parameters(), self.round_global.named_parameters())
            ):
                if name1 != name2:
                    raise RuntimeError

                else:
                    vec.append((param1 - param2).view(-1, 1))

            all_vec = torch.cat(vec)
            loss = 0.5 * self.global_alpha * torch.norm(all_vec)  # (all_vec** 2).sum().sqrt()

        else:
            raise NotImplemented

        return loss
