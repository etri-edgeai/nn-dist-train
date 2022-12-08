import torch
import torch.nn as nn
import numpy as np
import copy
import time
import os
import sys
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.fedntd.ClientTrainer import ClientTrainer
from algorithms.fedntd.criterion import *
from train_tools.measures import *

__all__ = ["Server"]


class Server:
    def __init__(
        self,
        model,
        data_distributed,
        optimizer,
        scheduler=None,
        evaluator=None,
        n_rounds=300,
        sample_ratio=0.1,
        local_epochs=5,
        device="cuda:1",
        algo_params=None,
        inline=False,
    ):
        """
        Server class controls the overall experiment.
        """
        self.num_classes = data_distributed["num_classes"]
        local_criterion = self._get_local_criterion(algo_params, self.num_classes)

        self.client = ClientTrainer(
            copy.deepcopy(model),
            local_epochs,
            device,
            local_criterion,
            self.num_classes,
        )
        self.model = model
        self.testloader = data_distributed["global"]["test"]
        self.criterion = nn.CrossEntropyLoss()
        self.data_distributed = data_distributed
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.scheduler = scheduler
        self.sample_ratio = sample_ratio
        self.n_rounds = n_rounds
        self.device = device
        self.n_clients = len(data_distributed["local"].keys())
        self.local_epochs = local_epochs
        self.server_results = {
            "client_history": [],
            "classwise_accuracy": [],
            "test_accuracy": [],
        }
        self.inline = inline
        print("\n>>> FedNTD Server initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        for round_idx in range(self.n_rounds):

            # Initial Model Statistics
            if round_idx == 0:
                classwise_acc, test_acc, _ = evaluate_model_classwise(
                    self.model, self.testloader, self.num_classes, device=self.device
                )
                self.server_results["classwise_accuracy"].append(classwise_acc)
                self.server_results["test_accuracy"].append(test_acc)

            start_time = time.time()

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            self.server_results["client_history"].append(sampled_clients)

            # (Distributed) global weights
            dg_weights = copy.deepcopy(self.model.state_dict())

            # Client training stage to upload weights & stats
            updated_local_weights, client_sizes, round_results = self._clients_training(
                sampled_clients
            )
            eval_results = None

            # Get aggregated weights & update global
            ag_weights = self._aggregation(updated_local_weights, client_sizes)

            # Update Global Server Model
            self.model.load_state_dict(ag_weights)

            # Measure Accuracy Statistics
            classwise_acc, test_acc, server_logits = evaluate_model_classwise(
                self.model,
                self.testloader,
                self.num_classes,
                get_logits=True,
                device=self.device,
            )
            self.server_results["classwise_accuracy"].append(classwise_acc)
            self.server_results["test_accuracy"].append(test_acc)
            self.server_results["logits_vec"] = server_logits  # Overwrite

            # Evaluator Phase
            round_weights = [dg_weights, updated_local_weights, ag_weights]
            eval_results = None

            if self.evaluator is not None:
                eval_results = self.evaluator.inspection(
                    round_weights, round_results, self.server_results
                )

            # Change learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            round_elapse = time.time() - start_time
            self._wandb_logging(round_results, round_idx, self.inline)

#             self._wandb_logging(round_results, eval_results, round_idx, self.inline)
            self._print_stats(round_results, test_acc, round_idx, round_elapse)
            print("-" * 50)

    def _clients_training(self, sampled_clients):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, client_sizes = [], []
        round_results = {}

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(server_weights, server_optimizer)

            # Local training
            local_results, local_size = self.client.train()

            # Upload locals
            updated_local_weights.append(self.client.upload_local())

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        return updated_local_weights, client_sizes, round_results

    def _client_sampling(self, round_idx):
        """Sample clients by given sampling ratio"""
        np.random.seed(
            round_idx
        )  # make sure for same client sampling for fair comparison
        clients_per_round = max(int(self.n_clients * self.sample_ratio), 1)
        sampled_clients = np.random.choice(
            self.n_clients, clients_per_round, replace=False
        )

        return sampled_clients

    def _set_client_data(self, client_idx):
        self.client.datasize = self.data_distributed["local"][client_idx]["datasize"]
        self.client.trainloader = self.data_distributed["local"][client_idx]["train"]
        self.client.testloader = self.data_distributed["global"]["test"]

    def _aggregation(self, w, ns):
        """Average locally trained model parameters"""
        prop = torch.tensor(ns, dtype=torch.float)
        prop /= torch.sum(prop)
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * prop[0]

        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * prop[i]

        return copy.deepcopy(w_avg)

    def _get_local_criterion(self, algo_params, num_classes):
        tau = algo_params.tau
        beta = algo_params.beta
        k = algo_params.k
        lam = algo_params.lam

        if algo_params.loss_type == "kd":
            criterion = KD_Loss(num_classes, tau, beta)
        elif algo_params.loss_type == "ntd":
            criterion = NTD_Loss(num_classes, tau, beta, k)
        elif algo_params.loss_type == "combine":
            criterion = Combine_Loss(num_classes, 3, 1, 1, lam)
        else:
            raise NotImplementedError

        return criterion

    def _results_updater(self, round_results, local_results):
        """Combine local results as clean format"""

        for key, item in local_results.items():
            if key not in round_results.keys():
                round_results[key] = [item]
            else:
                round_results[key].append(item)

        return round_results

    def _print_start(self):
        """Print initial log for experiment"""

        if self.device == "cpu":
            return "cpu"

        if isinstance(self.device, str):
            device_idx = int(self.device[-1])
        elif isinstance(self.device, torch._device):
            device_idx = self.device.index

        device_name = torch.cuda.get_device_name(device_idx)
        print("")
        print("=" * 50)
        print("Train start on device: {}".format(device_name))
        print("=" * 50)

    def _print_stats(self, round_results, test_accs, round_idx, round_elapse):
        print(
            "[Round {}/{}] Elapsed {}s (Current Time: {})".format(
                round_idx + 1,
                self.n_rounds,
                round(round_elapse, 1),
                time.strftime("%H:%M:%S"),
            )
        )
        print(
            "[Local Stat (Train Acc)]: {}, Avg - {:2.2f} (std {:2.2f})".format(
                round_results["train_acc"],
                np.mean(round_results["train_acc"]),
                np.std(round_results["train_acc"]),
            )
        )

        print(
            "[Local Stat (Test Acc)]: {}, Avg - {:2.2f} (std {:2.2f})".format(
                round_results["test_acc"],
                np.mean(round_results["test_acc"]),
                np.std(round_results["test_acc"]),
            )
        )

        print("[Server Stat] Acc - {:2.2f}".format(test_accs))
        
        
    def _wandb_logging(self, round_results, round_idx, inline=False):
        """Log on the W&B server"""

        if inline:
            return

        else:
            # Local round results
            local_results = {
                "local_train_acc": np.mean(round_results["train_acc"]),
                "local_test_acc": np.mean(round_results["test_acc"]),
            }
            wandb.log(local_results, step=round_idx) #각 ensemble의 train, test accuracy를 단순 averaging한 것!!

            # Server round results
            server_results = {
                "server_test_acc": self.server_results["test_accuracy"][-1]
            }
            wandb.log(server_results, step=round_idx) # 업데이트된 모델의 test accuracy!!
        
        

#     def _wandb_logging(self, round_results, eval_results, round_idx, inline=False):
#         """Log on the W&B server"""

#         if inline:
#             return

#         else:
#             # Local round results
#             local_results = {
#                 "local_train_acc": np.mean(round_results["train_acc"]),
#                 "local_test_acc": np.mean(round_results["test_acc"]),
#             }
#             wandb.log(local_results, step=round_idx)

#             # Server round results
#             server_results = {
#                 "server_test_acc": self.server_results["test_accuracy"][-1]
#             }
#             wandb.log(server_results, step=round_idx)
#             wandb.log(eval_results, step=round_idx)
