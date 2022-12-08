import torch
import torch.nn.functional as F
from .utils import *

import numpy as np

__all__ = ["calc_div", "calc_in_accs", "calc_wnorms", "calc_mean_var"]


def calc_div(cwa, local_dist_list):
    lg_div_list = []
    norm_cwa = F.normalize(cwa, dim=0, p=1)
    for local_dist_vec in local_dist_list:
        local_dist_vec = torch.Tensor(local_dist_vec)
        div_value = (torch.sqrt(((local_dist_vec - norm_cwa) ** 2).sum())).item()
        # div_value = (1 - torch.sqrt(local_dist_vec * norm_cwa).sum()).item()
        # div_value = F.kl_div(local_dist_vec, norm_cwa, reduction="batchmean").item()
        lg_div_list.append(div_value)

    return lg_div_list


def calc_in_accs(local_cwa_list, local_dist_list):
    in_acc_list = []

    for local_cwa, local_dist_vec in zip(local_cwa_list, local_dist_list):
        local_dist_vec = torch.Tensor(local_dist_vec)
        in_acc = torch.dot(local_cwa, local_dist_vec).item()
        in_acc_list.append(in_acc)

    return in_acc_list


def calc_wnorms(dg_weights, l_weights_list):
    dg2l_norm_list, l2l_norm_list = [], []

    # Flatten all weights
    dg_flat = flatten_weights(dg_weights)
    l_flat_list = []

    for l_weights in l_weights_list:
        l_flat_list.append(flatten_weights(l_weights))

    # DG vs L
    for l_flat in l_flat_list:
        dg2l_norm = calc_norm(dg_flat, l_flat)
        dg2l_norm_list.append(dg2l_norm)

    return dg2l_norm_list


def calc_mean_var(lg_div_list, in_acc_list, wnorm_list):
    mean_var_dict = {}
    mean_var_dict["lg_div_mean"] = np.mean(lg_div_list)
    mean_var_dict["in_acc_mean"] = np.mean(in_acc_list)
    mean_var_dict["wnorm_mean"] = np.mean(wnorm_list)

    mean_var_dict["lg_div_var"] = np.std(lg_div_list)
    mean_var_dict["in_acc_var"] = np.std(in_acc_list)
    mean_var_dict["w_norm_var"] = np.std(wnorm_list)

    return mean_var_dict


def flatten_weights(weights):
    """
    Flattens a PyTorch model state_dict. i.e., concat all parameters as a single, large vector.
    """
    all_params = []
    for param in weights.values():
        all_params.append(param.view(-1))
    all_params = torch.cat(all_params)

    return all_params


def calc_norm(weights1, weights2):
    w_norm = (weights1.cpu() - weights2.cpu()).norm().item()

    return w_norm
