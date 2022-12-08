import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

__all__ = ["scenario_features", "alignment_analyzer"]


def cwf_getter(model, testloader, device="cuda:1"):
    model.eval()
    model.to(device)

    all_targets, all_features = None, None

    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.to(device), targets.to(device)
            logits, features = model(data, get_features=True)
            all_targets = tensor_concater(all_targets, targets.cpu())
            all_features = tensor_concater(all_features, features.cpu())

        classwise_features = []

        for i in range(10):
            classwise_features.append(all_features[all_targets == i])

        cwf = torch.stack(classwise_features)
        cwf = torch.mean(cwf, dim=1)
        cwf = cwf.max(dim=0)[1].view(1, -1)

    return cwf


def local_trainer(model, local_loader, device="cuda:1"):
    model.train()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for _ in range(5):
        for data, targets in local_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


def aggregation(w, ns):
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


def tensor_concater(tensor1, tensor2, device=None):
    if tensor1 is None:
        tensor1 = tensor2

    else:
        if device is not None:
            tensor1 = tensor1.to(device)
            tensor2 = tensor2.to(device)

        tensor1 = torch.cat((tensor1, tensor2), dim=0)

    return tensor1.to(device)


def scenario_features(model, data_dict, sampled_clients=None, device="cuda:1"):
    testloader = data_dict["global"]["test"]
    dg_weights = copy.deepcopy(model.state_dict())

    dg_cwf = cwf_getter(model, testloader, device)
    l_cwf_list = []

    l_weight_list, l_size_list = [], []

    for client_idx in sampled_clients:
        model.load_state_dict(dg_weights)
        local_loader = data_dict["local"][client_idx]["train"]
        local_trainer(model, local_loader, device)

        l_weight = copy.deepcopy(model.state_dict())
        l_weight_list.append(l_weight)

        l_size = data_dict["local"][client_idx]["datasize"]
        l_size_list.append(l_size)

        l_cwf = cwf_getter(model, testloader, device)
        l_cwf_list.append(l_cwf)

    ag_weights = aggregation(l_weight_list, l_size_list)
    model.load_state_dict(ag_weights)
    ag_cwf = cwf_getter(model, testloader, device)

    return dg_cwf, l_cwf_list, ag_cwf


def elem_analyzer1(elem_single, elem_list):
    a = []
    for i in range(len(elem_list)):
        value = (elem_single == elem_list[i]).sum() / elem_single.size(1)
        a.append(value)

    mean, std = np.mean(a), np.std(a)

    return mean, std


def elem_analyzer2(elem_list1, elem_list2):
    a = []
    for i in range(len(elem_list1)):
        for j in range(len(elem_list2)):
            value = (elem_list1[i] == elem_list2[j]).sum() / elem_list1[0].size(1)
            a.append(value)

    mean, std = np.mean(a), np.std(a)

    return mean, std


def alignment_analyzer(dg_cwf, l_cwf_list, ag_cwf):
    result_dict = dict.fromkeys(["dg2l", "l2l", "l2ag", "dg2ag"])

    result_dict["dg2l"] = elem_analyzer1(dg_cwf, l_cwf_list)
    result_dict["l2ag"] = elem_analyzer1(ag_cwf, l_cwf_list)
    result_dict["l2l"] = elem_analyzer2(l_cwf_list, l_cwf_list)
    result_dict["dg2ag"] = (dg_cwf == ag_cwf).sum() / dg_cwf.size(1)

    return result_dict
