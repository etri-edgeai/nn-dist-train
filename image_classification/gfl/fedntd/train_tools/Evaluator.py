import torch
import numpy as np
import copy

from .measures import *

__all__ = ["Evaluator"]


class Evaluator:
    """Evaluate statistics for analysis"""

    def __init__(self, model, data_distributed, eval_params, scenario):
        self.model = copy.deepcopy(model)
        self.data_distributed = data_distributed
        self.eval_params = eval_params
        self.n_clients = len(data_distributed["local"].keys())
        self.best_acc_memory = [0 for i in range(self.n_clients)]
        self.n_rounds = scenario.n_rounds
        self.device = scenario.device

    def inspection(self, round_weights, round_results, server_results):
        """Conduct inspection on selected measures"""
        eval_results = {}
        testloader = self.data_distributed["global"]["test"]

        # Prediction Evaluation
        if self.eval_params.predictions: #True
            cwa_list = server_results["classwise_accuracy"]
            cl_f_dict = cl_forgetting_measure(cwa_list)
            dict_concater(eval_results, cl_f_dict)

            sampled_clients = server_results["client_history"][-1]
            local_dist_list, local_size_list = sampled_clients_identifier(
                self.data_distributed, sampled_clients
            )

            dg_cwa = server_results["classwise_accuracy"][-2]
            local_cwa_list = round_results["classwise_accuracy"]
            ag_cwa = server_results["classwise_accuracy"][-1]

            sp_dict = sp_analyzer(
                dg_cwa, local_cwa_list, ag_cwa, local_dist_list, local_size_list
            )
            dict_concater(eval_results, sp_dict)

            # server_logits = server_results["logits_vec"]
            # local_logits_list = round_results["logits_vec"]
            # a_dict = alignment_analyzer(
            #    server_logits, local_logits_list, testloader, local_size_list
            # )
            # dict_concater(eval_results, a_dict)

        if self.eval_params.weights: #False
            weights_results = inspect_weights(round_weights)

        if self.eval_params.features: #False
            pass

        #         #        if self.eval_params.lg_div:
        #         dg_cwa = server_results["classwise_accuracy"][-2]
        #         local_cwa_list = round_results["classwise_accuracy"]
        #         local_dist_list, _ = sampled_clients_identifier(
        #             self.data_distributed, sampled_clients
        #         )

        #         # Get KL-Div (l_dist, dg_cwa)
        #         lg_div_list = calc_div(dg_cwa, local_dist_list)
        #         eval_results["lg_div"] = np.array(lg_div_list)

        #         # Get accuracy statistics
        #         in_acc_list = calc_in_accs(local_cwa_list, local_dist_list)
        #         eval_results["in_acc"] = np.array(in_acc_list)

        #         # Get weight norm statitics
        #         dg_weights, l_weights_list, _ = round_weights
        #         wnorm_list = calc_wnorms(dg_weights, l_weights_list)
        #         eval_results["wnorm"] = np.array(wnorm_list)

        #         # Get variance statistics
        #         mean_var_dict = calc_mean_var(lg_div_list, in_acc_list, wnorm_list)
        #         dict_concater(eval_results, mean_var_dict)

        #         # Get Scatter tables
        #         # div_acc_table = get_scatter_table(
        #         #    lg_div_list, in_acc_list, key1="lg_div", key2="in_acc"
        #         # )
        #         # div_norm_table = get_scatter_table(
        #         #    lg_div_list, wnorm_list, key1="lg_div", key2="wnorm"
        #         # )
        #         eval_results["lg_div_list"] = lg_div_list
        #         eval_results["in_acc_list"] = in_acc_list
        #         eval_results["wnorm_list"] = wnorm_list

        return eval_results
