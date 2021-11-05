import torch
from torch.utils.data import DataLoader

from local import *
from utils import *
from train_tools import *

import numpy as np
import wandb, copy, time

__all__ = ["Server"]


class Server:
    def __init__(self):


    def train(self):
        return

    def _clients_training(self):
        return

    def _client_sampler(self):
        return

    def _total_results_updater(self):
        return

    def _print_stat(self, total_results, round_results, fed_round, round_elapse):
        print(
            "[Round {}/{}] Elapsed {}s/it".format(
                fed_round + 1, self.n_rounds, round(round_elapse, 1)
            )
        )
        print(
            "[Local Stat (Train Acc)]: {}, Avg - {:2.2f}".format(
                round_results["train_acc"], total_results["avg_train_acc"][fed_round]
            )
        )
        print(
            "[Local Stat (Test Acc)]: {}, Avg - {:2.2f}".format(
                round_results["test_acc"], total_results["avg_test_acc"][fed_round]
            )
        )
        print(
            "[Server Stat] Loss - {:.4f}, Acc - {:2.2f}".format(
                total_results["test_loss"][fed_round],
                total_results["test_acc"][fed_round],
            )
        )
        print("-" * 50)
