import torch
import torch.nn.functional as F
from .utils import *
import sys
__all__ = [
    "cl_forgetting_measure",
    "sp_analyzer",
    "alignment_analyzer",
    "evaluate_model",
    "evaluate_model_classwise",
]


def cl_forgetting_measure(cwa_list):
    """Calculate Continual-Learning view forgetting measure (BWT)"""
    a = torch.stack(cwa_list)
    a_max = torch.max(a, dim=0)[0]
    f = (a_max - a[-1]).mean().item()
    cl_f_dict = {"cl_f": f}
    return cl_f_dict


def sp_analyzer(dg_cwa, local_cwa_list, ag_cwa, local_dist_list, local_size_list):
    sp_dict = {}

    strict_stability_list, stability_list, plasticity_list = [], [], []
    out_dist_acc_list, in_dist_acc_list = [], []
    dg_out_dist_acc_list, dg_in_dist_acc_list = [], []
    ag_out_dist_acc_list, ag_in_dist_acc_list = [], []
    local_size_prop = F.normalize(torch.Tensor(local_size_list), dim=0, p=1)

    for local_cwa, local_dist_vec in zip(local_cwa_list, local_dist_list):
        # Stability (Out-dist info loss)
        local_dist_vec = torch.Tensor(local_dist_vec)
        strict_inverse_dist_vec = calculate_strict_inverse_dist(local_dist_vec)
        inverse_dist_vec = calculate_inverse_dist(local_dist_vec)
        cwa_drop = dg_cwa - local_cwa
        strict_stability = torch.dot(cwa_drop, strict_inverse_dist_vec)
        strict_stability_list.append(strict_stability)
        stability = torch.dot(cwa_drop, inverse_dist_vec)
        stability_list.append(stability)
        out_dist_acc = torch.dot(local_cwa, inverse_dist_vec)
        out_dist_acc_list.append(out_dist_acc)

        # Plasticity (In-dist info gain)
        cwa_gain = local_cwa - dg_cwa
        plasticity = torch.dot(cwa_gain, local_dist_vec)
        plasticity_list.append(plasticity)
        in_dist_acc = torch.dot(local_cwa, local_dist_vec)
        in_dist_acc_list.append(in_dist_acc)

        dg_out_dist_acc_list.append(torch.dot(dg_cwa, inverse_dist_vec))
        ag_out_dist_acc_list.append(torch.dot(ag_cwa, inverse_dist_vec))
        dg_in_dist_acc_list.append(torch.dot(dg_cwa, local_dist_vec))
        ag_in_dist_acc_list.append(torch.dot(ag_cwa, local_dist_vec))

    round_strict_stability = torch.Tensor(strict_stability_list)
    round_stability = torch.Tensor(stability_list)
    round_plasticity = torch.Tensor(plasticity_list)
    round_out_dist_acc = torch.Tensor(out_dist_acc_list)
    round_in_dist_acc = torch.Tensor(in_dist_acc_list)

    sp_dict["out_dist_acc"] = torch.dot(round_out_dist_acc, local_size_prop).item()
    sp_dict["in_dist_acc"] = torch.dot(round_in_dist_acc, local_size_prop).item()
    sp_dict["stability"] = torch.dot(round_stability, local_size_prop).item()
    sp_dict["strict_stability"] = torch.dot(
        round_strict_stability, local_size_prop
    ).item()
    sp_dict["plasticity"] = torch.dot(round_plasticity, local_size_prop).item()
    sp_dict["dg_out_dist_acc"] = torch.dot(
        torch.Tensor(dg_out_dist_acc_list), local_size_prop
    ).item()
    sp_dict["ag_out_dist_acc"] = torch.dot(
        torch.Tensor(ag_out_dist_acc_list), local_size_prop
    ).item()
    sp_dict["dg_in_dist_acc"] = torch.dot(
        torch.Tensor(dg_in_dist_acc_list), local_size_prop
    ).item()
    sp_dict["ag_in_dist_acc"] = torch.dot(
        torch.Tensor(ag_in_dist_acc_list), local_size_prop
    ).item()

    return sp_dict


def alignment_analyzer(server_logits, local_logits_list, dataloader, local_size_list):
    a_dict = {"align_pred": 0.0, "align_drop": 0.0, "ensemble_acc": 0.0}

    all_targets = None

    for _, targets in dataloader:
        all_targets = tensor_concater(all_targets, targets, device="cpu")

    count = all_targets.size(0)
    local_sizes = torch.Tensor(local_size_list).view(-1, 1, 1)

    server_pred = server_logits.max(dim=1)[1]
    server_correct = (server_pred == all_targets).long()
    server_acc = server_correct.sum().item() / count

    ensemble_logits = torch.stack(local_logits_list)
    ensemble_logits = (ensemble_logits * local_sizes).mean(dim=0)
    ensemble_pred = ensemble_logits.max(dim=1)[1]
    ensemble_correct = (ensemble_pred == all_targets).long()
    ensemble_acc = ensemble_correct.sum().item() / count

    a_dict["align_pred"] = (server_pred == ensemble_pred).sum().item() / count
    a_dict["align_drop"] = ensemble_acc - server_acc
    a_dict["ensemble_acc"] = ensemble_acc

    return a_dict


def calculate_inverse_dist(dist_vec):
    inverse_dist_vec = (1 - dist_vec) / (dist_vec.nelement() - 1)

    return inverse_dist_vec


def calculate_strict_inverse_dist(dist_vec):
    out_class_num = (dist_vec == 0).sum()
    inverse_dist_vec = (dist_vec == 0).float() / out_class_num

    return inverse_dist_vec


def evaluate_model(model, dataloader, device="cuda:1", get_features=False): #train acc 분석용
    model.eval()
    model.to(device)

    features_vec = None

    running_count = 0
    running_correct = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)

            if get_features:#False
                logits, features = model(data, get_features=get_features)
                features_vec = tensor_concater(features_vec, features, device="cpu")
            else:
                logits = model(data)

            pred = logits.max(dim=1)[1]

            running_correct += (targets == pred).sum().item()
            running_count += data.size(0)

    accuracy = round(running_correct / running_count, 4)

    if get_features: #False
        return accuracy, features_vec
    else:
        return accuracy


def evaluate_model_classwise(
    model,
    dataloader,
    num_classes=10,
    get_logits=False,
    get_features=False,
    device="cuda:1",
): #test acc 분석용

    model.eval()#평가하고자 하는 model
    model.to(device)

    classwise_count = torch.Tensor([0 for _ in range(num_classes)]).to(device)
    classwise_correct = torch.Tensor([0 for _ in range(num_classes)]).to(device)

    features_vec = None
    logits_vec = None

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            if get_features:#없어도 됨!!
                logits, features = model(data, get_features=get_features)
                features_vec = tensor_concater(features_vec, features, device="cpu")
            else:
                logits = model(data)
            
            preds = logits.max(dim=1)[1]#[0]은 value [1]은 index 뽑아냄
            if get_logits:#없어도 됨!!
                logits_vec = tensor_concater(logits_vec, logits)

            for class_idx in range(num_classes):
                class_elem = (targets == class_idx) #batch 내에서 target이 class_idx에 해당하는지 여부에 대한 boolean!!
                classwise_count[class_idx] += class_elem.sum().item()
                classwise_correct[class_idx] += (
                    (targets == preds)[class_elem].sum().item()
                )

        classwise_accuracy = classwise_correct / classwise_count
        accuracy = round(classwise_accuracy.mean().item(), 4)#각 class별로 갯수 같으니까 simple mean해도 상관 없음!!
    if logits_vec is not None:#없어도 됨!!
        logits_vec = logits_vec.cpu()

    if get_features:#없어도 됨!!
        return classwise_accuracy.cpu(), accuracy, logits_vec, features_vec

    else:
        return classwise_accuracy.cpu(), accuracy, logits_vec
