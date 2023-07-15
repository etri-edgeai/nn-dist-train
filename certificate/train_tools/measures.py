import torch, copy
import numpy as np
from scipy.special import softmax
from .measure_utils import *

__all__ = [
    "compute_kl_divergence",
    "compute_js_divergence",
    "calc_l2_norm",
    "get_variance",
    "calc_cka",
    "calc_class_cka",
    "calc_per_class_cka",
]


def calc_class_cka(per_class_cka, dist_vector1, dist_vector2, calc_type="out"):
    num_classes = len(per_class_cka)

    if dist_vector1 == "global":
        dist_vector1 = torch.Tensor([1 / num_classes for x in range(num_classes)])

    if dist_vector2 == "global":
        dist_vector2 = torch.Tensor([1 / num_classes for x in range(num_classes)])

    if calc_type == "in":
        in_cka1 = sum(per_class_cka[c] * dist_vector1[c] for c in range(num_classes))
        in_cka2 = sum(per_class_cka[c] * dist_vector2[c] for c in range(num_classes))
        in_cka = (in_cka1 + in_cka2) / 2
        return in_cka.item()

    elif calc_type == "out":
        reverse_vec1 = reverse_prob(dist_vector1)
        reverse_vec2 = reverse_prob(dist_vector2)

        out_cka1 = sum(per_class_cka[c] * reverse_vec1[c] for c in range(num_classes))
        out_cka2 = sum(per_class_cka[c] * reverse_vec1[c] for c in range(num_classes))

        out_cka = (out_cka1 + out_cka2) / 2
        return out_cka.item()


def calc_per_class_cka(rep1, rep2, labels, kernel="linear", num_classes=10):
    class_rep1 = dict.fromkeys(range(num_classes))
    class_rep2 = dict.fromkeys(range(num_classes))

    for c in range(num_classes):
        rep1_c, rep2_c = rep1[labels == c], rep2[labels == c]
        class_rep1[c] = gram_linear(rep1_c)
        class_rep2[c] = gram_linear(rep2_c)

    per_class_cka = []

    for c in range(num_classes):
        cka_class_element = cka(class_rep1[c], class_rep2[c])
        per_class_cka.append(cka_class_element)

    return per_class_cka


def reverse_prob(dist_vector):
    reverse_dist = 1 - dist_vector
    reverse_dist = reverse_dist / reverse_dist.sum()

    return reverse_dist


def calc_cka(rep1, rep2, kernel="linear"):
    rep1, rep2 = gram_linear(rep1), gram_linear(rep2)
    return cka(rep1, rep2).item()


def compute_kl_divergence(p_logits, q_logits):
    """ "KL (p || q)"""
    p_probs = softmax(p_logits, axis=1)
    q_probs = softmax(q_logits, axis=1)

    kl_div = p_probs * np.log(p_probs / q_probs + 1e-12)
    return np.mean(np.sum(kl_div, axis=1), axis=0)


def compute_js_divergence(p_logits, q_logits):
    p_probs = softmax(p_logits, axis=1)
    q_probs = softmax(q_logits, axis=1)
    m = 0.5 * (p_probs + q_probs)

    kld_p_m = np.sum(p_probs * np.log(p_probs / m + 1e-12), axis=1)
    kld_q_m = np.sum(q_probs * np.log(q_probs / m + 1e-12), axis=1)
    js = np.sqrt(0.5 * (kld_p_m + kld_q_m))
    return float(np.mean(js, axis=0))


def calc_delta(local, server):
    """local - server"""
    delta = {}
    for k, v in server:
        delta[k] = local[k] - v

    return delta


def get_delta_params(server, locals):
    delta_locals = []
    for local in locals:
        delta_locals.append(calc_delta(local, server))
    return delta_locals


def calc_var(server, local):
    _var = {}
    for k in server:
        _var[k] = torch.matmul(
            server[k].reshape((-1, 1)).T, local[k].reshape((-1, 1))
        ).item()
    return _var


def get_vec(order, vecs, device):
    ret = {}
    for k in order:
        serial_vec = torch.empty((1, 0)).to(device)

        weight = vecs[f"{k}.weight"].reshape((1, -1))
        serial_vec = torch.cat((serial_vec, weight), axis=-1)

        if f"{k}.bias" in vecs.keys():
            bias = vecs[f"{k}.bias"].reshape((1, -1))
            serial_vec = torch.cat((serial_vec, bias), axis=-1)

        ret[k] = serial_vec
    return ret


def get_local_vec(order, locals, device):
    ret = []
    for local in locals:
        ret.append(get_vec(order, local, device))
    return ret


def get_variance(calc_order, server_state_dict, locals, device):
    _vars = []
    server_vecs = get_vec(calc_order, server_state_dict, device)
    local_vecs = get_local_vec(calc_order, locals, device)

    cos = torch.nn.CosineSimilarity(dim=1)
    ret_cos = {}

    for layer in calc_order:
        val_cos = torch.tensor([0], dtype=torch.float, device=device)
        for i, local_vec in enumerate(local_vecs):
            local_cos = torch.clamp(
                cos(server_vecs[layer], local_vec[layer]), max=1, min=-1
            )
            val_cos += torch.abs(torch.acos(local_cos))
        val_cos /= len(local_vecs)
        ret_cos[layer] = round(val_cos.item(), 3)

    return ret_cos


def calc_l2_norm(order, state_dict, device):
    ret = get_vec(order, state_dict, device)

    for k in ret.keys():
        ret[k] = torch.norm(ret[k]).item()

    return ret
