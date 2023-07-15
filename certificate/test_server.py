# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader

torch.set_printoptions(10)

from server import *
from local import *
from train_tools import *
from train_tools.utils import model_evaluator
from train_tools.criterion import OverhaulLoss
from utils import *
from train_tools.models.layer3 import Layer3
from torch.utils.data import DataLoader

import os, random
import warnings

warnings.filterwarnings("ignore")


    
##############################################################################################
def _get_setups():
    # train setups
    model = Layer3()

    criterion = OverhaulLoss(**{"mode": "CE",
                                "num_classes": 10})

    return model, criterion


################################################################################################
def main():
    # Setups
    model, criterion = _get_setups()

    device = "cpu"
    testset = torch.load('./data/testset.pth')
    dataloader = DataLoader(testset, batch_size=128)
    
    server_weight = torch.load('./checkpoint/server_ckpt.pth')
    
    model.load_state_dict(server_weight)

    test_loss, test_acc = model_evaluator(model, dataloader, criterion, device)
    print(f'test loss : {test_loss}, test_acc : {test_acc}')
    return test_loss, test_acc


if __name__ == "__main__":
    test_loss, test_acc = main()
