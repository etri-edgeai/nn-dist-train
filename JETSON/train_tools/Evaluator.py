import torch
import torch.nn as nn
import torch.nn.functional as F

from .measure_utils import *
from .measures import *
from .utils import *

__all__ = ["Evaluator"]


class Evaluator:
    """Evaluate statistics for analysis"""

    def __init__(
        self, cka=None, global_analysis=None, in_inter_angle=None, device="cuda:0"
    ):
        self.cka = cka
        self.global_analysis = global_analysis
        self.in_inter_angle = in_inter_angle
        self.device = device
        self.test_correct = None

    def evaluation(
        self,
        model,
        test_loader,
        dist_global,
        agg_global,
        up_locals,
        local_identities,
        fed_round=None,
    ):
        eval_result = {}

        if fed_round == 0:
            return eval_result

        # Evaluation on feature angles
        if self.in_inter_angle.enabled:
            (
                dg_in,
                dg_inter,
                l_in,
                l_inter,
                l_inset_in,
                l_inset_inter,
                l_outset_in,
                l_outset_inter,
                stdev_list,
            ) = feature_angle_inspector(
                model,
                test_loader,
                model.state_dict(),
                up_locals,
                local_identities,
                device="cuda:0",
            )

            (
                l_in_std,
                l_inter_std,
                l_inset_in_std,
                l_inset_inter_std,
                l_outset_in_std,
                l_outset_inter_std,
            ) = stdev_list

            in_inter_angle = {
                "dg_in": dg_in,
                "dg_inter": dg_inter,
                "l_in": l_in,
                "l_inter": l_inter,
                "l_inset_in": l_inset_in,
                "l_inset_inter": l_inset_inter,
                "l_outset_in": l_outset_in,
                "l_outset_inter": l_outset_inter,
            }

            in_inter_angle_stdev = {
                "l_in_std": l_in_std,
                "l_inter_std": l_inter_std,
                "l_inset_in_std": l_inset_in_std,
                "l_inset_inter_std": l_inset_inter_std,
                "l_outset_in_std": l_outset_in_std,
                "l_outset_inter_std": l_outset_inter_std,
            }

        # Evaluation on CKA
        if (self.cka.enabled) and (fed_round % self.cka.period == 0):
            cka_gl, cka_ll, cka_lg, cka_gg = cka_evaluator(
                model,
                test_loader,
                dist_global=model.state_dict(),
                agg_global=agg_global,
                up_locals=up_locals,
                mode=self.cka.mode,
                device=self.device,
            )

            cka_result = {
                "[CKA] DG -> L": cka_gl,
                "[CKA] L vs L": cka_ll,
                "[CKA] L -> AG": cka_lg,
                "[CKA] DG -> AG": cka_gg,
            }

            eval_result["cka"] = cka_result

        return eval_result
        
    def global_evaluation(
        self, model, clients_dataset, fed_round, test_loader=None, mode="forward"
    ):
        """Evaluation on prediction consistency"""

        running_correct, running_size = 0, 0
        if self.global_analysis.enabled:
            model.to(self.device)
            with torch.no_grad():
                for local_set in clients_dataset:
                    dataloader = torch.utils.data.DataLoader(local_set, batch_size=50)
                    for data, target in dataloader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        running_size += data.size(0)
                        running_correct += (output.max(dim=1)[1] == target).sum().item()

                gl_acc = round((running_correct / running_size) * 100, 2)

            if mode == "forward":
                pass

            elif mode == "backward":

                correct_tensor = None

                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    correct = output.max(dim=1)[1] == target
                    correct_tensor = tensor_concater(correct_tensor, correct)

                if self.test_correct is not None:
                    global_analysis = {
                        "mutual_correct": ((self.test_correct + correct_tensor) == 2)
                        .sum()
                        .item(),
                        "mutual_incorrect": ((self.test_correct + correct_tensor) == 0)
                        .sum()
                        .item(),
                        "switch_to_incorrect": ((self.test_correct > correct_tensor))
                        .sum()
                        .item(),
                        "switch_to_correct": ((self.test_correct < correct_tensor))
                        .sum()
                        .item(),
                    }

                self.test_correct = correct_tensor.int()

