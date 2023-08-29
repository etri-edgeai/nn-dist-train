import torch
import torch.nn as nn
import torch.nn.functional as F

def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]#not true인 것만 추려낸다.
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits


class NTD_Loss(nn.Module):
    def __init__(self, num_classes=100, tau=3, w_ce=1, w_kd=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.w_ce = w_ce
        self.w_kd = w_kd

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = self.w_ce* ce_loss + self.w_kd * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        logits = refine_as_not_true(logits, targets, self.num_classes)
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)

        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed dg_model prediction
        with torch.no_grad():
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = self.KLDiv(pred_probs, dg_probs)

        return loss
    
    
class LSD_Loss(nn.Module):
    def __init__(self, num_classes=100, tau=3, w_ce=1, w_kd=1):
        super(LSD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.w_ce = w_ce
        self.w_kd = w_kd

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        lsd_loss = self._lsd_loss(logits, dg_logits, targets)

        loss = self.w_ce* ce_loss + self.w_kd * lsd_loss

        return loss

    def _lsd_loss(self, logits, dg_logits, targets):
        pred_probs = F.log_softmax(logits / self.tau, dim=1)
        
        # Get smoothed dg_model prediction
        with torch.no_grad():
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = self.KLDiv(pred_probs, dg_probs)

        return loss
    
class MSE_Loss(nn.Module):
    def __init__(self, num_classes=100, tau=3, w_ce=1, w_kd=1):
        super(MSE_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.w_ce = w_ce
        self.w_kd = w_kd

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        mse_loss = self.MSE(logits, dg_logits)

        loss = self.w_ce* ce_loss + self.w_kd * mse_loss

        return loss
    
    
    
class NEW_Loss(nn.Module):
    def __init__(self, num_classes=100, tau=2, w_ce=1, w_kd=9):
        super(NEW_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.w_ce = w_ce
        self.w_kd = w_kd

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        lsd_loss = self._lsd_loss(logits, dg_logits, targets)

        loss = self.w_ce* ce_loss + self.w_kd * lsd_loss

        return loss

    def _lsd_loss(self, logits, dg_logits, targets):
        pred_probs = F.log_softmax(logits * self.tau, dim=1)
        
        # Get smoothed dg_model prediction
        with torch.no_grad():
            dg_probs = torch.softmax(dg_logits * self.tau, dim=1)

        loss = self.KLDiv(pred_probs, dg_probs)

        return loss    

