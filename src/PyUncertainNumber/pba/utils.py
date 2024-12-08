import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .interval import Interval
from .intervalOperators import wc_interval, make_vec_interval
from collections import namedtuple
from .interval import Interval as nInterval

cdf_bundle = namedtuple('cdf_bundle', ['quantile', 'probability'])


def stacking(vec_interval: list[nInterval | Interval], weights, display=False):
    """ stochastic mixture operation for DS structure and Intervals 

    args:
        - l_un (list): list of uncertain numbers
        - weights (list): list of weights
        - display (Boolean): boolean for plotting

    return:
        - the left and right bounds in respective tuples
    """

    vec_interval = make_vec_interval(vec_interval)

    q1, p1 = weighted_ecdf(vec_interval.lo, weights)
    q2, p2 = weighted_ecdf(vec_interval.hi, weights)

    if display:
        fig, ax = plt.subplots()
        ax.step(q1, p1, marker='+', c='g', where='post')
        ax.step(q2, p2, marker='+', c='b', where='post')
        ax.plot([q1[0], q2[0]], [0, 0], c='g')
        ax.plot([q1[-1], q2[-1]], [1, 1], c='b')
    return cdf_bundle(q1, p1), cdf_bundle(q2, p2)


def sorting(list1, list2):
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return list1, list2


def weighted_ecdf(s, w=None, display=False):
    """ compute the weighted ecdf from (precise) sample data 

    note:
        - Sudret eq.1
    """

    if w is None:
        # weights
        N = len(s)
        w = np.repeat(1/N, N)

    s, w = sorting(s, w)
    p = np.cumsum(w)

    # for box plotting
    q = np.insert(s, 0, s[0], axis=0)
    p = np.insert(p, 0, 0., axis=0)

    if display == True:
        fig, ax = plt.subplots()
        ax.step(q, p, marker='+', where='post')

    # return quantile and probabilities
    return q, p


def reweighting(*masses):
    """ reweight the masses to sum to 1 """
    masses = np.ravel(masses)
    return masses / masses.sum()


def round():
    pass


def uniform_reparameterisation(a, b):
    """ reparameterise the uniform distribution to a, b """
    a, b = wc_interval(a), wc_interval(b)
    return a, b-a


def find_nearest(array, value):
    """ find the index of the nearest value in the array to the given value """

    array = np.asarray(array)
    # find the nearest value
    ind = (np.abs(array - value)).argmin()
    return ind


@mpl.rc_context({"text.usetex": True})
def plot_intervals(vec_interval: list[nInterval | Interval], ax=None, **kwargs):
    # TODO finish the codes as this is temporary
    """ 

    args:
        vec_interval: vectorised interval objects
    """
    vec_interval = make_vec_interval(vec_interval)

    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    for i, intl in enumerate(vec_interval):
        # horizontally plot the interval
        ax.plot([intl.lo, intl.hi], [i, i], **kwargs)
    return ax


@mpl.rc_context({"text.usetex": True})
def plot_DS_structure(vec_interval, weights, ax=None, **kwargs):
    ax = plot_intervals(vec_interval, ax=ax, **kwargs)

    # add the weights after each interval element
    for i, interval in enumerate(vec_interval):
        ax.text(interval.hi() + 0.3,
                i,
                f"{weights[i]:.2f}",
                verticalalignment='center',
                horizontalalignment='right')
    ax.margins(x=0.2, y=0.2)
    ax.set_yticks([])
    ax.set_title('Dempster Shafer structures')
    return ax


def _interval_list_to_array(l, left=True):
    if left:
        def f(x): return x.left if isinstance(x, Interval) else x
    else:  # must be right
        def f(x): return x.right if isinstance(x, Interval) else x

    return np.array([f(i) for i in l])


def read_json(file_name):
    f = open(file_name)
    data = json.load(f)
    return data


def check_increasing(arr):
    return np.all(np.diff(arr) >= 0)


class NotIncreasingError(Exception):
    pass
