import torch

__all__ = ["refine_as_not_true", "get_tnt1_scale", "refine_as_not_true_positions"]


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
    logits = refine_as_not_true(logits, targets, num_classes)
    logits = logits[positions]

    return logits


def get_tnt1_scale(logits, targets, num_classes):
    t_logits = logits[range(logits.size(0)), targets]
    t_scale = t_logits.mean()

    nt_logits, _ = refine_as_not_true(logits, targets, num_classes)
    nt_logits1 = nt_logits.max(dim=1)[0]
    nt_scale = nt_logits1.mean()

    return t_scale, nt_scale
