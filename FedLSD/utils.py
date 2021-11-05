import wandb, os
import numpy as np

__all__ = ["wandb_logger", "calc_avg", "directory_setter", "objectview"]


def wandb_logger(total_results, round_results, fed_round, angles=None):
    wandb.log(
        {
            "test_loss": total_results["test_loss"][fed_round],
            "test_acc": total_results["test_acc"][fed_round],
            "avg_train_loss": total_results["avg_train_loss"][fed_round],
            "avg_train_acc": total_results["avg_train_acc"][fed_round],
            "avg_test_loss": total_results["avg_test_loss"][fed_round],
            "avg_test_acc": total_results["avg_test_acc"][fed_round],
            "train_acc(local)": wandb.Histogram(
                np_histogram=np.histogram(round_results["train_acc"])
            ),
            "test_acc(local)": wandb.Histogram(
                np_histogram=np.histogram(round_results["test_acc"])
            ),
            "fed_round": fed_round,
        },
        step=fed_round,
    )


def calc_avg(inputs, weights=None):
    if weights is None:
        return round(sum(inputs) / len(inputs), 4)

    else:
        products = [
            elem * (weight / sum(weights)) for elem, weight in zip(inputs, weights)
        ]
        return round(sum(products), 4)


def directory_setter(path="./results", make_dir=False):
    if not os.path.exists(path) and make_dir:
        os.makedirs(path)  # make dir if not exist
        print("directory %s is created" % path)

    if not os.path.isdir(path):
        raise NotADirectoryError(
            "%s is not valid. set make_dir=True to make dir." % path
        )


class objectview:
    def __init__(self, config):
        for k, v in config.items():
            if isinstance(v, dict):
                self.__dict__[k] = objectview(v)
            else:
                self.__dict__[k] = v
