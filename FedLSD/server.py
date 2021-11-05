import torch
from torch.utils.data import DataLoader

from local import *
from utils import *
from train_tools import *

import numpy as np
import wandb, copy, time

__all__ = ["Server"]


class Server:
    def __init__(
        self,
        datasets,
        model,
        criterion,
        optimizer,
        evaluator,
        scheduler=None,
        exp_name=None,
        local_args=None,
        n_rounds=300,
        n_clients=100,
        sample_ratio=0.1,
        agg_alg="fedavg",
        server_location="cpu",
        device="cuda:0",
    ):

        if local_args is not None:
            self.locals = Local(
                copy.deepcopy(model),
                criterion,
                datasets["test"],
                local_args.local_ep,
                local_args.local_bs,
                global_loss=[local_args.global_loss, local_args.global_alpha],
                server_location=server_location,
                device=device,
            )
        else:
            # make local as default setups
            self.locals = Local(
                copy.deepcopy(model),
                criterion,
                datasets["test"],
                server_location=server_location,
                device=device,
            )

        self.model = model
        self.exp_name = exp_name
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.scheduler = scheduler

        self.agg_alg = get_agg_alg(agg_alg)

        self.n_rounds = n_rounds
        self.n_clients = n_clients
        self.sample_ratio = sample_ratio
        self.local_sets = datasets["local"]
        self.test_loader = DataLoader(datasets["test"], batch_size=256)

        self.server_location = server_location
        self.device = device

    def train(self):
        return

    def _clients_training(self):
        """Conduct local training and get trained local models' weights"""

        updated_locals = []
        round_results = {
            "local_sizes": [],
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for local_id, local_set in zip(sampled_clients, clients_dataset):

            # Download global
            self.locals.download_global(local_set, server_weights, server_optimizer)

            # Local training
            local_results = self.locals.train()

            # Upload locals
            updated_locals.append(self.locals.upload_local())

            # Update results
            round_results["train_loss"].append(local_results["train_loss"])
            round_results["train_acc"].append(local_results["train_acc"])
            round_results["test_loss"].append(local_results["test_loss"])
            round_results["test_acc"].append(local_results["test_acc"])
            round_results["local_sizes"].append(local_set.__len__())
            round_results["local_results"] = local_results

            # Reset local model
            self.locals.reset()

        return updated_locals, round_results

    def _client_sampler(self):
        """Sample clients by given sampling ratio"""
        clients_per_round = max(int(self.sample_ratio * self.n_clients), 1)
        sampled_clients = np.random.choice(
            self.n_clients, clients_per_round, replace=False
        )
        return sampled_clients

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
