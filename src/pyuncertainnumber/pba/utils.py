from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .intervalOperators import wc_interval, make_vec_interval
from collections import namedtuple
from .interval import Interval as nInterval
from dataclasses import dataclass
from .intervals import Interval

cdf_bundle = namedtuple("cdf_bundle", ["quantiles", "probabilities"])
""" a handy composition object for a c.d.f which is a tuple of quantile and probability 
#TODO I mean `ecdf_bundle` essentially, but changing name may introduce compatibility issues now
#TODO to be deprecated
note:
    - handy to represent bounding c.d.fs for pbox, especially for free-form pbox
"""


@dataclass
class CDF_bundle:
    quantiles: np.ndarray
    probabilities: np.ndarray
    # TODO plot ecdf not starting from 0

    @classmethod
    def from_sps_ecdf(cls, e):
        """utility to tranform sps.ecdf to cdf_bundle"""
        return cls(e.cdf.quantiles, e.cdf.probabilities)


def transform_ecdf_bundle(e):
    """utility to tranform sps.ecdf to cdf_bundle"""
    return CDF_bundle(e.cdf.quantiles, e.cdf.probabilities)


def pl_ecdf_bounding_bundles(
    b_l: CDF_bundle, b_r: CDF_bundle, alpha=0.025, ax=None, legend=True, title=None
):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(
        b_l.quantiles,
        b_l.probabilities,
        label="upper bound",
        drawstyle="steps-post",
        color="g",
    )
    ax.plot(
        b_r.quantiles,
        b_r.probabilities,
        label="lower bound",
        drawstyle="steps-post",
        color="b",
    )
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend()


def sorting(list1, list2):
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return list1, list2


def weighted_ecdf(s, w=None, display=False):
    """compute the weighted ecdf from (precise) sample data

    note:
        - Sudret eq.1
    """

    if w is None:
        # weights
        N = len(s)
        w = np.repeat(1 / N, N)

    s, w = sorting(s, w)
    p = np.cumsum(w)

    # for box plotting
    q = np.insert(s, 0, s[0], axis=0)
    p = np.insert(p, 0, 0.0, axis=0)

    if display == True:
        fig, ax = plt.subplots()
        ax.step(q, p, marker="+", where="post")

    # return quantile and probabilities
    return q, p


def reweighting(*masses):
    """reweight the masses to sum to 1"""
    masses = np.ravel(masses)
    return masses / masses.sum()


def round():
    pass


def uniform_reparameterisation(a, b):
    """reparameterise the uniform distribution to a, b"""
    #! incorrect in the case of Interval args
    a, b = wc_interval(a), wc_interval(b)
    return a, b - a


def find_nearest(array, value):
    """find the index of the nearest value in the array to the given value"""

    array = np.asarray(array)
    # find the nearest value
    ind = (np.abs(array - value)).argmin()
    return ind


@mpl.rc_context({"text.usetex": True})
def plot_intervals(vec_interval: list[nInterval | Interval], ax=None, **kwargs):
    """plot the intervals in a vectorised form
    args:
        vec_interval: vectorised interval objects
    """
    vec_interval = make_vec_interval(vec_interval)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    for i, intl in enumerate(vec_interval):  # horizontally plot the interval
        ax.plot([intl.lo, intl.hi], [i, i], **kwargs)
    ax.margins(x=0.1, y=0.1)
    ax.set_yticks([])
    return ax


def _interval_list_to_array(l, left=True):
    if left:

        def f(x):
            return x.left if isinstance(x, Interval) else x

    else:  # must be right

        def f(x):
            return x.right if isinstance(x, Interval) else x

    return np.array([f(i) for i in l])


def read_json(file_name):
    f = open(file_name)
    data = json.load(f)
    return data


def check_increasing(arr):
    return np.all(np.diff(arr) >= 0)


class NotIncreasingError(Exception):
    pass
