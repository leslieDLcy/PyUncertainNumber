from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .intervalOperators import wc_scalar_interval, make_vec_interval
from collections import namedtuple
from dataclasses import dataclass
from .intervals.number import Interval

cdf_bundle = namedtuple("cdf_bundle", ["quantiles", "probabilities"])
""" a handy composition object for a c.d.f which is a tuple of quantile and probability 
#TODO I mean `ecdf_bundle` essentially, but changing name may introduce compatibility issues now
#TODO to be deprecated
note:
    - handy to represent bounding c.d.fs for pbox, especially for free-form pbox
"""


@dataclass
class CDF_bundle:
    """a handy tuple of q and p for a CDF"""

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
    b_l: CDF_bundle, b_r: CDF_bundle, ax=None, legend=True, title=None
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


def weighted_ecdf(s, w=None, display=False) -> tuple:
    """compute the weighted ecdf from (precise) sample data

    args:
        s (array-like) : precise sample data
        w (array-like) : weights

    note:
        - Sudret eq.1

    return:
        ecdf in the form of a tuple of q and p
    """

    if w is None:
        # weights
        N = len(s)
        w = np.repeat(1 / N, N)
    else:
        w = np.array(w)

    # s, w = sorting(s, w)
    arr = np.stack((s, w), axis=1)
    arr = arr[np.argsort(arr[:, 0])]

    p = np.cumsum(arr[:, 1])

    # for box plotting
    q = np.insert(arr[:, 0], 0, arr[0, 0], axis=0)
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


def uniform_reparameterisation(a, b):
    """reparameterise the uniform distribution to a, b"""
    #! incorrect in the case of Interval args
    a, b = wc_scalar_interval(a), wc_scalar_interval(b)
    return a, b - a


# def find_nearest(array, value):
#     """find the index of the nearest value in the array to the given value

#     note:
#         it works both for quantiles and probabilities

#     return: scalar or vector.
#     """

#     array = np.asarray(array)
#     # find the nearest value
#     ind = (np.abs(array - value)).argmin()
#     return ind


import numpy as np


# def find_nearest(array, value):
#     """Find the index of the nearest value in the array to the given value(s).

#     Works with both scalar and array inputs for `value`.

#     Parameters:
#         array (np.ndarray): The array to search.
#         value (float or np.ndarray): The value(s) to find the nearest to.

#     Returns:
#         int or np.ndarray: Index or array of indices of nearest values.
#     """
#     array = np.asarray(array)

#     if np.isscalar(value):
#         # scalar case
#         ind = (np.abs(array - value)).argmin()
#         return ind
#     else:
#         # vectorized version for array input
#         value = np.asarray(value)
#         diff = np.abs(array[:, None] - value[None, :])
#         indices = np.argmin(diff, axis=0)
#         return indices


# TODO to test this high-performance version below
def find_nearest(array, value):
    """Find index/indices of nearest value(s) in `array` to each `value`.

    Efficient for both scalar and array inputs.
    """
    array = np.asarray(array)
    value = np.atleast_1d(value)

    # Compute distances using broadcasting
    diff = np.abs(array[None, :] - value[:, None])

    # Find index of minimum difference along axis 1
    indices = np.argmin(diff, axis=1)

    # Return scalar if input was scalar
    return indices[0] if np.isscalar(value) else indices


@mpl.rc_context({"text.usetex": True})
def plot_intervals(vec_interval: list[Interval], ax=None, **kwargs):
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


def is_increasing(arr):
    """check if 'arr' is increasing"""
    return np.all(np.diff(arr) >= 0)


class NotIncreasingError(Exception):
    pass


def condensation(bound, number):
    """a joint implementation for condensation"""
    if isinstance(bound, list | tuple):
        return condensation_bounds(bound, number)
    else:
        return condensation_bound(bound, number)


def condensation_bounds(bounds, number):
    """condense the bounds of number pbox

    args:
        number (int) : the number to be reduced
        bounds (list or tuple): the left and right bound to be reduced
    """
    b = bounds[0]

    if number > len(b):
        raise ValueError("Cannot sample more elements than exist in the list.")
    if len(bounds[0]) != len(bounds[1]):
        raise Exception("steps of two bounds are different")

    indices = np.linspace(0, len(b) - 1, number, dtype=int)

    l = np.array([bounds[0][i] for i in indices])
    r = np.array([bounds[1][i] for i in indices])
    return l, r


def condensation_bound(bound, number):
    """condense the bounds of number pbox

    args:
        number (int) : the number to be reduced
        bound (array-like): either the left or right bound to be reduced
    """

    if number > len(bound):
        raise ValueError("Cannot sample more elements than exist in the list.")

    indices = np.linspace(0, len(bound) - 1, number, dtype=int)

    new_bound = np.array([bound[i] for i in indices])
    return new_bound


def smooth_condensation(bounds, number=200):

    def smooth_ecdf(V, steps):

        m = len(V) - 1

        if m == 0:
            return np.repeat(V, steps)
        if steps == 1:
            return np.array([min(V), max(V)])

        d = 1 / m
        n = round(d * steps * 200)

        if n == 0:
            c = V
        else:
            c = []
            for i in range(m):
                v = V[i]
                w = V[i + 1]
                c.extend(np.linspace(start=v, stop=w, num=n))

        u = [c[round((len(c) - 1) * (k + 0) / (steps - 1))] for k in range(steps)]

        return np.array(u)

    l_smooth = smooth_ecdf(bounds[0], number)
    r_smooth = smooth_ecdf(bounds[1], number)
    return l_smooth, r_smooth


def equi_selection(arr, n):
    """draw n equidistant points from the array"""
    indices = np.linspace(0, len(arr) - 1, n, dtype=int)
    selected = arr[indices]
    return selected
