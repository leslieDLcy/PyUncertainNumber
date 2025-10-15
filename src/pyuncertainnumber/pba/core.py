from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike
import scipy.stats as sps
from numbers import Number
from pyuncertainnumber.pba.pbox_abc import Pbox, Staircase


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


def am_distance_counter(A, B):
    """Compute the distance between two sets of intervals as used in the area metric.

    notes:

    It is essentially doing:

    def f(a, b, c, d):
        return np.maximum.reduce([c - b, a - d, 0])

    """
    a, b = A[:, 0], A[:, 1]
    c, d = B[:, 0], B[:, 1]
    return np.maximum.reduce([c - b, a - d, np.zeros_like(a)])


def function_succeeds(f, *args, **kwargs):
    try:
        f(*args, **kwargs)
        return True
    except Exception:
        return False


def area_metric_pbox(a: Pbox, b: Pbox):
    """when a and b are both Pboxes"""
    # diff = endpoint_distance(a.to_numpy(), b.to_numpy())  # old version busted by Scott
    diff = am_distance_counter(a.to_numpy(), b.to_numpy())
    return np.trapz(y=diff, x=a.p_values)


def area_metric_sample(a: ArrayLike, b: ArrayLike):
    return sps.wasserstein_distance(a, b)


def area_metric_number(a: Pbox | Number, b: Pbox | Number) -> float:
    """if any of a or b is a number, compute area metric accordingly"""
    from pyuncertainnumber import pba

    if isinstance(a, Number) and isinstance(b, Number):
        return abs(a - b)
    if isinstance(a, Number):
        a, b = b, a  # swap so b is the number
    if isinstance(a, Pbox) and a.degenerate:
        return area_metric_ecdf(a.left, b, a.p_values)
    if isinstance(a, Pbox) and (not a.degenerate):
        # make b a Pbox
        b = pba.I(b).to_pbox()
        return area_metric_pbox(a, b)


def area_metric(a: Number | Pbox | ArrayLike, b: Number | Pbox | ArrayLike) -> float:
    """Compute the area metric between two objects.

    note:
        top-level function to compute area metric between any two objects
    """
    if isinstance(a, Number) or isinstance(b, Number):
        return area_metric_number(a, b)
    if isinstance(a, Pbox) and isinstance(b, Pbox):
        if a.degenerate and b.degenerate:
            return area_metric_ecdf(a.left, b.left, a.p_values)
        elif function_succeeds(a.imp, b):
            return 0.0
        else:
            return area_metric_pbox(a, b)
    if isinstance(a, (np.ndarray, list)) and isinstance(b, (np.ndarray, list)):
        return area_metric_sample(a, b)
    elif (isinstance(a, Pbox) and isinstance(b, np.ndarray)) or (
        isinstance(a, np.ndarray) and isinstance(b, Pbox)
    ):
        # make a a Pbox and b a sample anyway
        if not isinstance(a, Pbox):
            a, b = b, a

        # b has to be a scalar sample
        b = np.squeeze(b).item()
        return area_metric_number(a, b)
    else:
        raise NotImplementedError("Area metric not implemented for these types.")


from bisect import bisect_left


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
