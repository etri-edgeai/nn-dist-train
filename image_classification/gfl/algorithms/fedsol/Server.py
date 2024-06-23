import copy
import time
import os, sys


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.fedsol.ClientTrainer import ClientTrainer
from algorithms.BaseServer import BaseServer
from algorithms.measures import *

from train_tools.preprocessing.cifar10.loader import get_dataloader_cifar10
from train_tools.preprocessing.cifar100.loader import get_dataloader_cifar100

__all__ = ["Server"]

DATA_LOADERS = {
    "cifar10": get_dataloader_cifar10,
    "cifar100": get_dataloader_cifar100,
}


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        """
        Server class controls the overall experiment.
        """
        self.client = ClientTrainer(
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

        self.rho = self.algo_params["rho"]

        print("\n>>> FedSOL (Fixed Rho) Server initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        for round_idx in range(self.n_rounds):

            start_time = time.time()

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            self.server_results["client_history"].append(sampled_clients)

            # Client training stage to upload weights & stats
            updated_local_weights, client_sizes, round_results = self._clients_training(
                sampled_clients
            )

            # Get aggregated weights & update global
            ag_weights = self._aggregation(updated_local_weights, client_sizes)

            self._update_and_evaluate(ag_weights, round_results, round_idx, start_time)

            # if (round_idx + 1) % 100 == 0:
            #     self._save_server_model(round_idx)

    def _clients_training(self, sampled_clients, finetune=False):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, client_sizes = [], []
        round_local_results = {}

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:

            # Fetch client datasets
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(server_weights, server_optimizer)

            # Local training
            local_results, local_size = self.client.train()

            # Upload locals
            updated_local_weights.append(self.client.upload_local())
            client_sizes.append(local_size)

            # Update results
            round_local_results = self._results_updater(
                round_local_results, local_results
            )

            # Reset local model
            self.client.reset()

        return updated_local_weights, client_sizes, round_local_results
