import torch

__all__ = ["refine_as_not_true", "refine_as_not_true_positions"]


def refine_as_not_true(logits, targets, num_classes, k=-1):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    if k == -1:
        logits, nt_topk_positions = logits.topk(k=num_classes - 1, dim=1)

    else:
        logits, nt_topk_positions = logits.topk(k=k, dim=1)

    return logits, nt_topk_positions


def refine_as_not_true_positions(logits, targets, num_classes, positions):
    nt_logits, _ = refine_as_not_true(logits, targets, num_classes)
    nt_refined = None

    for slice_idx in range(nt_logits.size(0)):
        nt_refined_elem = nt_logits[slice_idx, positions[slice_idx]].view(1, -1)
        nt_refined = tensor_concater(nt_refined, nt_refined_elem)

    return nt_refined


def tensor_concater(tensor1, tensor2, device=None):
    if tensor1 is None:
        tensor1 = tensor2

    else:
        if device is not None:
            tensor1 = tensor1.to(device)
            tensor2 = tensor2.to(device)

        tensor1 = torch.cat((tensor1, tensor2), dim=0)

    return tensor1.to(device)
