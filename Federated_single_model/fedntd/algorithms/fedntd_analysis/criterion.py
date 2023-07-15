import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

__all__ = ["LSD_Loss", "NTD_Loss"]


class LSD_Loss(nn.Module):
    def __init__(self, num_classes=10, tau=3, beta=1):
        super(LSD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)

        # MSE loss on logits
        if self.tau == -1:
            mse_loss = self.MSE(logits, dg_logits)
            loss = ce_loss + self.beta * mse_loss

        else:
            lsd_loss = self._lsd_loss(logits, dg_logits)
            loss = ce_loss + self.beta * lsd_loss

        return loss

    def _lsd_loss(self, logits, dg_logits):
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed dg_model prediction
        with torch.no_grad():
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = self.KLDiv(pred_probs, dg_probs)

        return loss


class NTD_Loss(nn.Module):
    def __init__(self, num_classes=10, tau=3, beta=1, k=-1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta
        self.k = k

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)

        # MSE loss on logits
        if self.tau == -1:
            nt_mse_loss = self._nt_mse_loss(logits, dg_logits, targets)
            loss = ce_loss + self.beta * nt_mse_loss

        else:
            ntd_loss = self._ntd_loss(logits, dg_logits, targets)
            loss = ce_loss + self.beta * ntd_loss

        return loss

    def _nt_mse_loss(self, logits, dg_logits, targets):
        logits, nt_topk_positions = refine_as_not_true(
            logits, targets, self.num_classes, k=self.k
        )
        # dg_logits = refine_as_not_true_positions(
        #    dg_logits, targets, self.num_classes, positions=nt_topk_positions
        # )
        dg_logits, _ = refine_as_not_true(
            dg_logits, targets, self.num_classes, k=self.k
        )
        loss = self.MSE(logits, dg_logits)

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        logits, nt_topk_positions = refine_as_not_true(
            logits, targets, self.num_classes, k=self.k
        )
        dg_logits, _ = refine_as_not_true(
            dg_logits, targets, self.num_classes, k=self.k
        )

        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed dg_model prediction
        with torch.no_grad():
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = self.KLDiv(pred_probs, dg_probs)

        return loss
