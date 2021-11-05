import torch, copy
import numpy as np

__all__ = ["get_agg_alg"]


def get_agg_alg(alg):

    if "fedavg" == alg:

        def aggregate_models(w, ns):
            prop = torch.tensor(ns, dtype=torch.float)
            prop /= torch.sum(prop)
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                w_avg[k] = w_avg[k] * prop[0]

            for k in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[k] += w[i][k] * prop[i]
                # w_avg[k] = torch.div(w_avg[k],  torch.sum(prop))
                # w_avg[k] = torch.div(w_avg[k], len(w))
            return copy.deepcopy(w_avg)

        return aggregate_models

    else:
        raise NotImplementedError
