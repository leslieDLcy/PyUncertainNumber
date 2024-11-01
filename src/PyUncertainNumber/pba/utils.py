import json
import numpy as np
import matplotlib.pyplot as plt
from .interval import Interval


def plot_intervals(interval_list, ax=None, **kwargs):
    # TODO finish the codes as this is temporary
    """ 
    
    args:
        interval_list: list of Interval objects
    """
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    for i, intl in enumerate(interval_list):
        # horizontally plot the interval
        ax.plot([intl.left, intl.right], [i,i], **kwargs)
    
    return ax
    


def plot_DS_structure(interval_list, weights, ax=None, **kwargs):
    ax = plot_intervals(interval_list, ax=ax, **kwargs)

    # add the weights after each interval element
    for i in range(len(interval_list)):
        ax.text(interval_list[i].right + 0.3, 
                i, 
                f"{weights[i]:.2f}", 
                verticalalignment='center', 
                horizontalalignment='right')
    return ax




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
