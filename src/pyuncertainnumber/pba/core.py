from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
import scipy.stats as sps
from numbers import Number
from pyuncertainnumber.pba.pbox_abc import Pbox, Staircase
from pyuncertainnumber.pba.intervals import Interval
from bisect import bisect_left
import numpy as np
import matplotlib.pyplot as plt


class Joint(ABC):

    def __init__(self, copula, marginals: list):
        self.copula = copula
        self.marginals = marginals


def wasserstein_1d(q1: ArrayLike, q2: ArrayLike, p: ArrayLike) -> float:
    """An intuitive of Wasserstein metric in 1D, aka. area between two quantile functions

    This is equivaluent to the Area Metric in 1D, which shall return same results as "scipy.stats.wasserstein_distance"

    args:
        q1, q2 (ArrayLike): quantile vectors (same length, corresponding to probabilities p)

        p      (ArrayLike): probability vector (between 0 and 1, monotone increasing)
    """

    diff = np.abs(q1 - q2)
    return np.trapz(y=diff, x=p)


def area_metric_ecdf(q1, q2, p):
    """Wasserstein metric in 1D, aka. area between two quantile functions

    This is equivaluent to the Area Metric in 1D.

    args:
        q1, q2 (ArrayLike): quantile vectors (same length, corresponding to probabilities p)

        p      (ArrayLike): probability vector (between 0 and 1, monotone increasing).
                            Must be the same for q1 and q2
    """
    p = np.asarray(p)
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)

    diff = np.abs(q1 - q2)  # broadcasts if q1 or q2 is scalar
    if diff.shape != p.shape:
        # allow (scalar) -> expand to match p
        if diff.ndim == 0:
            diff = np.full_like(p, diff, dtype=float)
        else:
            raise ValueError("q1 and q2 must be broadcastable to the shape of p.")
    return np.trapz(y=diff, x=p)


def endpoint_distance(A, B):
    """
    Smallest endpoint distance elementwise between intervals.
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    if A.shape != B.shape:
        raise ValueError(
            "For elementwise comparison, A and B must have the same shape."
        )

    # compute all 4 endpoint differences per pair
    diffs = np.abs(A[:, :, None] - B[:, None, :])  # shape (n,2,2)
    distances = diffs.min(axis=(1, 2))  # shape (n,)

    if len(distances) == 1:
        return distances.item()
    return distances


# def plot_intervals_from_res(res, p, *, show_points=False, title=None, ax=None):
#     """
#     For each row where res['distance'] != 0, plot a horizontal segment from
#     min(x1, x2) to max(x1, x2) at height p[i].

#     Args
#     ----
#     res : structured np.ndarray with fields ('distance','x1','x2')
#     p   : 1D array-like of same length as res
#     show_points : if True, also plot small markers at the endpoints
#     title : optional figure title
#     """
#     res = np.asarray(res)
#     p = np.asarray(p)
#     if res.shape[0] != p.shape[0]:
#         raise ValueError("p must have the same number of rows as res")

#     mask, intervals = intervals_from_res(res)
#     if not np.any(mask):
#         # nothing to plot
#         fig, ax = plt.subplots()
#         ax.set_xlabel("x")
#         ax.set_ylabel("p")
#         if title:
#             ax.set_title(title)
#         return ax

#     y = p[mask]
#     lo = intervals[:, 0]
#     hi = intervals[:, 1]

#     if ax is None:
#         fig, ax = plt.subplots()
#     for xi0, xi1, yi in zip(lo, hi, y):
#         ax.plot([xi0, xi1], [yi, yi])  # horizontal segment
#         if show_points:
#             ax.plot([xi0, xi1], [yi, yi], marker="o", linestyle="")

#     ax.set_xlabel("x")
#     ax.set_ylabel("p (height)")
#     if title:
#         ax.set_title(title)
#     # Make a little padding around data
#     x_min = np.min(lo)
#     x_max = np.max(hi)
#     if x_min == x_max:
#         x_min -= 0.5
#         x_max += 0.5
#     ax.set_xlim(x_min, x_max)
#     # y padding
#     y_min = np.min(y)
#     y_max = np.max(y)
#     if y_min == y_max:
#         y_min -= 0.5
#         y_max += 0.5
#     ax.set_ylim(0, 1)

#     return ax


def area_metric_hint():
    pass


# * --------------------------- the developments below are between scalar and P-box only


def distance_to_ecdf_bound(x0, quantile):
    """Min horizontal distance from x0 to the ECDF defined by quantile."""
    xs = sorted(quantile)
    i = bisect_left(xs, x0)
    if i == 0:
        return abs(xs[0] - x0)
    if i == len(xs):
        return abs(x0 - xs[-1])
    # nearest of the two neighbors
    return min(abs(x0 - xs[i - 1]), abs(xs[i] - x0))


def closer_bound(x0, left_edge, right_edge):
    """Decide which ECDF bound is closer to x0.

    args:
        x0: a scalar point
        left_edge: samples from the left bound of the ECDF
        x_right_samples: samples from the right bound of the ECDF
    """
    dl = distance_to_ecdf_bound(x0, left_edge)
    dr = distance_to_ecdf_bound(x0, right_edge)
    if dl < dr:
        return "left", dl, dr
    if dr < dl:
        return "right", dl, dr
    return "tie", dl, dr  # exactly equidistant


def if_outside(x0, left_edge, right_edge):
    """Check if x0 is outside the ECDF defined by left_edge and right_edge."""
    return x0 < left_edge[0] or x0 > right_edge[-1]


def if_right_in(x0, left_edge, right_edge):
    """Check if x0 is inside the ECDF defined by left_edge and right_edge."""
    return left_edge[-1] <= x0 <= right_edge[0]


# high-level func
def directional(x0, left_edge, right_edge):
    """give instructions on which direction to move the Pbox towards the scalar

    returns:
        output a message variable
    """
    a = if_outside(x0, left_edge, right_edge)  # a is boolean
    msg, _, _ = closer_bound(x0, left_edge, right_edge)

    if a:  # outside
        if msg == "left" or msg == "tie":
            return "out_left"
        elif msg == "right":
            return "out_right"
    else:  # inside
        if msg == "left" or msg == "tie":
            return "in_left"
        elif msg == "right":
            return "in_right"


def calibration_distance(a: Pbox, b: Number) -> float:
    """Estimate the calibration distance to compensate area metric between a P-box aand a scalar"""

    if not isinstance(a, Pbox):
        a, b = b, a  # swap so that a is always the Pbox

    msg, _, _ = closer_bound(b, a.left, a.right)

    if msg == "left" or msg == "tie":
        return np.abs(a.left[-1] - b)
    elif msg == "right":
        return np.abs(a.right[0] - b)


def slide_pbox_towards_scalar(a, b):
    """Slide the Pbox a towards the scalar b by one step.

    args:
        a: a Pbox
        b: a scalar

    returns:
        a new Pbox that is slid towards b by one step
    """
    # which direction to slide
    msg = directional(b, a.left, a.right)

    # how much to slide
    proposal_dd = calibration_distance(a, b)

    # worked and backup
    # match msg:
    #     case "out_right":
    #         # expand on the right
    #         return Staircase(a.left, a.right + proposal_dd)
    #     case "out_left":
    #         # expand on the left
    #         return Staircase(a.left - proposal_dd, a.right)
    #     case "in_left":
    #         # slide to the left
    #         return Staircase(a.left - proposal_dd, a.right - proposal_dd)
    #     case "in_right":
    #         # slide to the right
    #         return Staircase(a.left + proposal_dd, a.right + proposal_dd)

    match msg:
        case "out_right":
            # expand on the right
            d1 = 0
            d2 = proposal_dd
        case "out_left":
            # expand on the left
            d1 = proposal_dd
            d2 = 0
        case "in_left":
            # slide to the left
            d1 = proposal_dd
            d2 = -proposal_dd
        case "in_right":
            # slide to the right
            d1 = -proposal_dd
            d2 = proposal_dd

    return Staircase(a.left - d1, a.right + d2)
