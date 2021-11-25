import torch
import torch.nn.functional as F
from .models import *
from .measures import *

import numpy as np
import copy, random

__all__ = [
    "create_nets",
    "feature_stacker",
    "cka_evaluator",
    "model_evaluator",
    "feature_angle_inspector",
    "tensor_concater",
    "locals_identifier",
    "identifier",
    "norm_g2l",
    "norm_l2l",
    "random_seeder",
]


MODELS = {
    "mnistcnn": MnistCNN,
    "cifarcnn": CifarCNN,
    "vgg11": vgg11,
    "res8": resnet8,
    "res14": resnet14,
    "res20": resnet20,
}


def create_nets(
    model, dataset="cifar10", location="cpu", num_classes=10, use_bias=True
):
    """Create a network model"""

    # Get image shape
    if "mnist" in dataset:
        dim_in = 1
        img_size = 1 * 28 * 28

    elif "cifar" in dataset:
        dim_in = 3
        img_size = 3 * 32 * 32

    # Build model architecture
    if "vgg" in model:
        model = MODELS[model](
            img_size=img_size, dim_in=dim_in, num_classes=num_classes, use_bias=use_bias
        )

    else:
        model = MODELS[model](dim_in=dim_in, num_classes=num_classes, use_bias=use_bias)

    return model


@torch.no_grad()
def feature_stacker(model, dataloader, device):
    model.to(device)
    stacked_features = None

    for data, _ in dataloader:
        data = data.to(device)

        _, features = model(data, get_features=True)
        stacked_features = tensor_concater(stacked_features, features)

    return stacked_features


@torch.no_grad()
def cka_evaluator(
    model,
    dataloader,
    dist_global=None,
    agg_global=None,
    up_locals=None,
    mode="ALL",
    device="cuda:0",
):
    """Measure CKA of models: (DG vs L), (L vs L), (L vs AG), (DG vs AG)"""
    cka_gl, cka_ll, cka_lg, cka_gg = 0, 0, 0, 0

    backbone = copy.deepcopy(model)
    backbone.to(device).eval()

    if dist_global is not None:
        backbone.load_state_dict(dist_global)
        dg_features = feature_stacker(backbone, dataloader, device)

    if agg_global is not None:
        backbone.load_state_dict(agg_global)
        ag_features = feature_stacker(backbone, dataloader, device)

    if up_locals is not None:
        ul_features = []
        for w in up_locals:
            backbone.load_state_dict(w)
            features = feature_stacker(backbone, dataloader, device)
            ul_features.append(features)

    if ("GL" in mode) or (mode == "ALL"):
        running_sum = 0
        for ul_f in ul_features:
            running_sum += calc_cka(dg_features, ul_f)
        cka_gl = round(running_sum / len(ul_features), 4)

    if ("LL" in mode) or (mode == "ALL"):
        running_sum = 0
        for ul_f_i in ul_features:
            for ul_f_j in ul_features:
                running_sum += calc_cka(ul_f_i, ul_f_j)
        cka_ll = round(running_sum / (len(ul_features) * len(ul_features)), 4)

    if ("LG" in mode) or (mode == "ALL"):
        running_sum = 0
        for ul_f in ul_features:
            running_sum += calc_cka(ag_features, ul_f)
        cka_lg = round(running_sum / len(ul_features), 4)

    if ("GG" in mode) or (mode == "ALL"):
        cka_gg = round(calc_cka(dg_features, ag_features), 4)

    else:
        raise NotImplementedError

    del dg_features, ag_features, ul_features

    return cka_gl, cka_ll, cka_lg, cka_gg


@torch.no_grad()
def model_evaluator(model, dataloader, criterion, device):
    running_loss, running_correct, data_num = 0, 0, 0
    stacked_features, stacked_targets = None, None

    model.to(device).eval()
    for itr, (data, target) in enumerate(dataloader):
        data_num += data.size(0)
        data, target = data.to(device), target.to(device)

        logits, features = model(data, get_features=True)
        pred = torch.max(logits, dim=1)[1]

        stacked_features = tensor_concater(stacked_features, features)
        stacked_labels = tensor_concater(stacked_targets, target)

        running_correct += (pred == target).sum().item()
        running_loss += F.cross_entropy(logits, target).item()

    eval_loss = round(running_loss / data_num, 4)
    eval_acc = round((running_correct / data_num) * 100, 2)
    eval_features = stacked_features.mean(dim=0)

    model.train()

    return eval_loss, eval_acc


@torch.no_grad()
def feature_angle_inspector(
    model, dataloader, dg_model, l_models, local_identities, device="cuda:0"
):
    labels = torch.LongTensor(dataloader.dataset.targets).to(device)

    backbone = copy.deepcopy(model)
    backbone.to(device).eval()

    # global model
    backbone.load_state_dict(dg_model)
    dg_features = feature_stacker(backbone, dataloader, device)
    dg_features = F.normalize(dg_features, dim=1)

    dg_class_features = features_by_classes(dg_features, labels)
    dg_in, dg_inter = in_inter_class_angle(dg_class_features)

    l_ins, l_inters = [], []
    l_inset_ins, l_inset_inters, l_outset_ins, l_outset_inters = [], [], [], []

    for i in range(len(l_models)):
        w, l_identity = l_models[i], local_identities[i]
        backbone.load_state_dict(w)
        features = feature_stacker(backbone, dataloader, device)
        features = F.normalize(features, dim=1)
        l_class_features = features_by_classes(features, labels)
        l_in, l_inter = in_inter_class_angle(l_class_features)
        l_ins.append(l_in)
        l_inters.append(l_inter)

        inset_features, outset_features = [], []
        for i in range(10):
            if i in l_identity:
                inset_features.append(l_class_features[i])
            else:
                outset_features.append(l_class_features[i])

        inset_in, inset_inter = in_inter_class_angle(inset_features)
        if inset_in is not None:
            l_inset_ins.append(inset_in)
        if inset_inter is not None:
            l_inset_inters.append(inset_inter)

        if len(outset_features) > 0:
            outset_in, outset_inter = in_inter_class_angle(outset_features)
            l_outset_ins.append(outset_in)
            l_outset_inters.append(outset_inter)

    l_in = np.array(l_ins).mean()
    l_inter = np.array(l_inters).mean()
    l_inset_in = np.array(l_inset_ins).mean()
    l_inset_inter = np.array(l_inset_inters).mean()
    l_in_std = np.array(l_ins).std()
    l_inter_std = np.array(l_inters).std()
    l_inset_in_std = np.array(l_inset_ins).std()
    l_inset_inter_std = np.array(l_inset_inters).std()

    if len(l_outset_ins) > 0:
        l_outset_in = np.array(l_outset_ins).mean()
        l_outset_inter = np.array(l_outset_inters).mean()
        l_outset_in_std = np.array(l_outset_ins).std()
        l_outset_inter_std = np.array(l_outset_inters).std()
    else:
        l_outset_in = -1
        l_outset_inter = -1
        l_outset_in_std = 0
        l_outset_inter_std = 0

    stdev_list = [
        l_in_std,
        l_inter_std,
        l_inset_in_std,
        l_inset_inter_std,
        l_outset_in_std,
        l_outset_inter_std,
    ]

    return (
        dg_in,
        dg_inter,
        l_in,
        l_inter,
        l_inset_in,
        l_inset_inter,
        l_outset_in,
        l_outset_inter,
        stdev_list,
    )

@torch.no_grad()
def in_inter_class_angle(feature_list):
    in_angles, inter_angles = [], []
    for i in range(len(feature_list)):
        for j in range(len(feature_list)):
            f1 = feature_list[i]
            f2 = feature_list[j]
            angle = torch.einsum('nk, ak -> na', f1, f2)
            if i == j:
                in_angles.append(angle.cpu())
            else:
                inter_angles.append(angle.cpu())
    
    if len(in_angles) > 1:
        in_angles = torch.cat(in_angles, dim=1)
        in_mean, in_std = in_angles.mean().item(), in_angles.std().item()
    else:
        in_mean = in_angles[0].mean().item()

    if len(inter_angles) > 1:
        inter_angles = torch.cat(inter_angles, dim=1)
        inter_mean, inter_std = inter_angles.mean().item(), inter_angles.std().item()
    elif len(inter_angles) == 1:
        inter_mean = inter_angles[0].mean().item()
    else:
        inter_mean = None

    return in_mean, inter_mean

def tensor_concater(tensor1, tensor2, device=None):
    if tensor1 is None:
        tensor1 = tensor2

    else:
        if device is not None:
            tensor1 = tensor1.to(device)
            tensor2 = tensor2.to(device)

        tensor1 = torch.cat((tensor1, tensor2), dim=0)

    return tensor1.to(device)


def locals_identifier(local_sets):
    targets_list = torch.Tensor(local_sets[0].dataset.targets).long()

    local_identities = {}
    for i in range(len(local_sets)):
        local_identities[i] = targets_list[local_sets[i].indices].unique().tolist()

    return local_identities


def identifier(local_set):
    targets_list = torch.Tensor(local_set.dataset.targets).long()
    local_identity = targets_list[local_set.indices].unique().tolist()
    return local_identity


def features_by_classes(features, labels):
    class_features_list = []
    for c in range(labels.unique().size(0)):
        class_features_list.append(features[labels == c])

    return class_features_list


@torch.no_grad()
def norm_g2l(global_weight, updated_locals, device="cuda:0"):
    norms = []

    g_flat = weight_flattener(global_weight)

    for l_w in updated_locals:
        l_flat = weight_flattener(l_w)
        w_norm = torch.norm(g_flat.to(device) - l_flat.to(device))
        norms.append(w_norm.item())

    norms = np.array(norms)

    return norms.mean(), norms.std()


@torch.no_grad()
def norm_l2l(updated_locals):
    norms = []

    for i, local_i in enumerate(updated_locals):
        for j, local_j in enumerate(updated_locals):
            if i != j:
                w1 = weight_flattener(local_i)
                w2 = weight_flattener(local_j)
                w_norm = torch.norm(w1 - w2)
                norms.append(w_norm.item())

    norms = np.array(norms)

    return norms.mean(), norms.std()


def weight_flattener(weight_dict):
    weight_flat = None

    for _, weight in weight_dict.items():
        weight_flat = tensor_concater(weight_flat, weight.view(-1))

    return weight_flat


def random_seeder(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
