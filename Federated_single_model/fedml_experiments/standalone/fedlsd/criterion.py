import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ["OverhaulLoss"]


class OverhaulLoss(nn.Module):
    """Loss function (Local Objective)"""

    def __init__(self, mode="CE", num_classes=10, temp=1, beta=1, lam=0):
        super(OverhaulLoss, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.temp = temp
        self.beta = beta
        self.lam = lam

    def forward(self, logits, target, t_logits=None):
        """Calculate loss by given modes"""

        loss = torch.zeros(logits.size(0)).to(str(target.device))  # initialize loss

        # Cross-Entropy Loss (FedAvg Baseline)
        if args.criterion_mode == "CE":
            loss += F.cross_entropy(logits / self.temp, target, reduction="none")

        # ------------------------------ Proposed Methods --------------------------------------- #
        elif args.criterion_mode == "LSD":
            # get DG(Distributed Global) model prediction
            with torch.no_grad():
                t_distill = torch.softmax(t_logits / self.temp, dim=1)

            # FedLSD Loss
            ce_loss = cross_entropy(logits, target, reduction="none")
            lsd_loss = (self.temp ** 2) * cross_entropy(
                logits / self.temp, t_distill, reduction="none"
            )

            loss = (1 - self.beta) * ce_loss + self.beta * lsd_loss

        elif args.criterion_mode == "LS-NTD":
            # get DG(Distributed Global) model prediction for not-true classes
            with torch.no_grad():
                hard_target = onehot(target, N=self.num_classes).float()
                nt_mask = hard_target != 1
                nt_t_logits = t_logits * nt_mask
                nt_t_logits[
                    nt_mask != 1
                    ] = -100000  # very small number to true class logits
                nt_distill = torch.softmax(nt_t_logits / self.temp, dim=1)

            ce_loss = cross_entropy(logits, target, reduction="none")

            # FedLS-NTD Loss
            nt_logits = logits * nt_mask  # get not-true class logits from local output
            kd_loss = (self.temp ** 2) * cross_entropy(
                nt_logits / self.temp, nt_distill, reduction="none"
            )

            loss = (1 - self.beta) * ce_loss + self.beta * kd_loss


        elif args.criterion_mode == "LSD_NTD":  # For analysis the effect of LSD vs. LS-NTD
            # get DG(Distributed Global) model prediction
            with torch.no_grad():
                t_distill = torch.softmax(t_logits / self.temp, dim=1)

            # get CE Loss
            ce_loss = cross_entropy(logits, target, reduction="none")

            # get LSD Loss
            lsd_loss = (self.temp ** 2) * cross_entropy(
                logits / self.temp, t_distill, reduction="none"
            )

            # get DG(Distributed Global) model prediction for not-true classes
            with torch.no_grad():
                hard_target = onehot(target, N=self.num_classes).float()
                nt_mask = hard_target != 1
                nt_t_logits = t_logits * nt_mask
                nt_t_logits[nt_mask != 1] = -100000
                nt_distill = torch.softmax(nt_t_logits / self.temp, dim=1)

            # get LS-NTD Loss
            nt_logits = logits * nt_mask
            ntd_loss = (self.temp ** 2) * cross_entropy(
                nt_logits / self.temp, nt_distill, reduction="none"
            )

            loss = (1 - self.beta) * ce_loss + self.beta * (
                    (1 - self.lam) * lsd_loss + self.lam * ntd_loss
            )
        # ------------------------------------------------------------------------------------------- #

        loss = loss.mean()

        return loss


### The code below is modified from : https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py
def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


def _is_long(x):
    if hasattr(x, "data"):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(
        inputs,
        target,
        weight=None,
        ignore_index=-100,
        reduction="mean",
        smooth_eps=None,
        smooth_dist=None,
        from_logits=True,
):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(
                inputs, target, weight, ignore_index=ignore_index, reduction=reduction
            )
        else:
            return F.nll_loss(
                inputs, target, weight, ignore_index=ignore_index, reduction=reduction
            )

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1.0 - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(
            self,
            weight=None,
            ignore_index=-100,
            reduction="mean",
            smooth_eps=None,
            smooth_dist=None,
            from_logits=True,
    ):
        super(CrossEntropyLoss, self).__init__(
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            smooth_eps=self.smooth_eps,
            smooth_dist=smooth_dist,
            from_logits=self.from_logits,
        )