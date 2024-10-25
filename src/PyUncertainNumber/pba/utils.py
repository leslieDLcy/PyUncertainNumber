import json
import numpy as np
from .interval import Interval

def _interval_list_to_array(l, left=True):
    if left:
        f = lambda x: x.left if isinstance(x, Interval) else x
    else:  # must be right
        f = lambda x: x.right if isinstance(x, Interval) else x

    return np.array([f(i) for i in l])


def read_json(file_name):
    f = open(file_name)
    data = json.load(f)
    return data


def check_increasing(arr):
    return np.all(np.diff(arr) >= 0)


class NotIncreasingError(Exception):
    pass
