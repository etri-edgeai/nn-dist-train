import torch
import torch.nn.functional as F

import numpy as np

__all__ = ["inspect_weights"]


def inspect_weights(round_weights):
    dg_weights, l_weights_list, ag_weights = round_weights
    # Flatten all weights
    dg_flat, ag_flat = flatten_weights(dg_weights), flatten_weights(ag_weights)
    l_flat_list = []

    for l_weights in l_weights_list:
        l_flat_list.append(flatten_weights(l_weights))

    # DG vs L
    dg2l_norm, dg2l_cos = [], []
    for l_flat in l_flat_list:
        dg2l_norm_elem, dg2l_cos_elem = calc_norm_cos(dg_flat, l_flat)
        dg2l_norm.append(dg2l_norm_elem)
        dg2l_cos.append(dg2l_cos_elem)

    dg2l_norm, dg2l_cos = np.mean(dg2l_norm), np.mean(dg2l_cos)

    # L vs L
    l2l_norm, l2l_cos = [], []
    for i in range(len(l_flat_list)):
        for j in range(len(l_flat_list)):
            if i != j:
                l2l_norm_elem, l2l_cos_elem = calc_norm_cos(
                    l_flat_list[i], l_flat_list[j]
                )
    l2l_norm.append(l2l_norm_elem)
    l2l_cos.append(l2l_cos_elem)
    l2l_norm, l2l_cos = np.mean(l2l_norm), np.mean(l2l_cos)

    # L vs AG
    l2ag_norm, l2ag_cos = [], []
    for l_flat in l_flat_list:
        l2ag_norm_elem, l2ag_cos_elem = calc_norm_cos(ag_flat, l_flat)
        l2ag_norm.append(l2ag_norm_elem)
        l2ag_cos.append(l2ag_cos_elem)

    l2ag_norm, l2ag_cos = np.mean(l2ag_norm), np.mean(l2ag_cos)

    # DG vs AG
    dg2ag_norm, dg2ag_cos = calc_norm_cos(dg_flat, ag_flat)

    # Combine all results
    result = {
        "dg2l_wnorm": dg2l_norm,
        "dg2l_wcos": dg2l_cos,
        "l2l_wnorm": l2l_norm,
        "l2l_wcos": l2l_cos,
        "l2ag_wnorm": l2ag_norm,
        "l2ag_wcos": l2ag_cos,
        "dg2ag_wnorm": dg2ag_norm,
        "dg2ag_wcos": dg2ag_cos,
    }

    for key, item in result.items():
        result[key] = round(item, 4)

    return result


def flatten_weights(weights):
    """
    Flattens a PyTorch model state_dict. i.e., concat all parameters as a single, large vector.
    """
    all_params = []
    for param in weights.values():
        all_params.append(param.view(-1))
    all_params = torch.cat(all_params)

    return all_params


def calc_norm_cos(weights1, weights2):
    w_norm = (weights1.cpu() - weights2.cpu()).norm().item()
    w_cos = F.cosine_similarity(weights1.cpu(), weights2.cpu(), dim=0).item()

    return w_norm, w_cos
