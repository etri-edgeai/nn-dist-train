import os
import numpy as np

__all__ = ["calc_avg", "directory_setter", "objectview"]


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
