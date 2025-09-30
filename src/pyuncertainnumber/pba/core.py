from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike
import scipy.stats as sps
from numbers import Number
from pyuncertainnumber.pba.pbox_abc import Pbox


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


def function_succeeds(f, *args, **kwargs):
    try:
        f(*args, **kwargs)
        return True
    except Exception:
        return False


def area_metric_pbox(a: Pbox, b: Pbox):
    """when a and b are both Pboxes"""
    diff = endpoint_distance(a.to_numpy(), b.to_numpy())
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
    else:
        raise NotImplementedError("Area metric not implemented for these types.")
