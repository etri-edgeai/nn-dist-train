import torch
import torch.nn as nn
import copy, time
import numpy as np

from .measures import *

__all__ = ["ClientTrainer", "Server"]


class ClientTrainer:
    def __init__(self, model, criterion, local_epochs, device, num_classes):
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        # model & optimizer
        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
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


class Server:
    def __init__(
        self, model, data_distributed, criterion, optimizer, local_epochs, device,
    ):
        self.num_classes = data_distributed["num_classes"]
        self.client = ClientTrainer(
            copy.deepcopy(model), criterion, local_epochs, device, self.num_classes
        )
        self.model = model
        self.testloader = data_distributed["global"]["test"]
        self.criterion = criterion
        self.data_distributed = data_distributed
        self.optimizer = optimizer
        self.device = device
        self.n_clients = len(data_distributed["local"].keys())
        self.local_epochs = local_epochs
        self.server_results = {
            "client_history": [],
            "classwise_accuracy": [],
            "test_accuracy": [],
        }

    def round_run(self, sampled_clients=None):
        """Run the FL experiment"""
        self._print_start()

        # Initial Model Statistics
        classwise_acc, test_acc, _ = evaluate_model_classwise(
            self.model, self.testloader, self.num_classes, device=self.device
        )
        self.server_results["classwise_accuracy"].append(classwise_acc)
        self.server_results["test_accuracy"].append(test_acc)

        start_time = time.time()

        self.server_results["client_history"].append(sampled_clients)

        # (Distributed) global weights
        dg_weights = copy.deepcopy(self.model.state_dict())

        # Client training stage to upload weights & stats
        updated_local_weights, client_sizes, round_results = self._clients_training(
            sampled_clients
        )

        # Get aggregated weights & update global
        ag_weights = self._aggregation(updated_local_weights, client_sizes)

        # Update Global Server Model
        self.model.load_state_dict(ag_weights)

        # Measure Accuracy Statistics
        classwise_acc, test_acc, _ = evaluate_model_classwise(
            self.model,
            self.testloader,
            self.num_classes,
            get_logits=False,
            device=self.device,
        )
        self.server_results["classwise_accuracy"].append(classwise_acc)
        self.server_results["test_accuracy"].append(test_acc)

        # Evaluator Phase
        round_weights = [dg_weights, updated_local_weights, ag_weights]

        round_elapse = time.time() - start_time

        self._print_stats(round_results, test_acc, 0, round_elapse)
        print("-" * 50)

        return round_weights

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
            "Round Elapsed {}s (Current Time: {})".format(
                round_idx + 1, round(round_elapse, 1), time.strftime("%H:%M:%S"),
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
