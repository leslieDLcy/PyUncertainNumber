"""leslie's general bound to bound implementation"""

from ...pba.intervals.number import Interval
from ...pba.intervalOperators import make_vec_interval

import numpy as np


def b2b(vecs, func, method=None, **kwargs) -> Interval:
    """
    General implementation of a function:

    Y = g(Ix1, Ix2, ..., IxN)

    where Ix1, Ix2, ..., IxN are intervals.

    In a general case, the function g is not necessarily monotonic and g() is a black-box model.
    Optimisation to the rescue and two of them particularly: GA and BO.

    args:
        vecs: list or tuple of scalar intervals
        func: performance or response function or a black-box model as in subprocess.
        method: the method used for interval propagation
            - 'endpoints': only the endpoints
            - 'ga': genetic algorithm
            - 'bo': bayesian optimisation
        *args: additional arguments to be passed to the function
        **kwargs: additional keyword arguments to be passed to the function

    signature:
        This shall be a top-level func as `epistemic_propagation()`.

    returns:
        Interval: the low and upper bound of the response
    """
    from pyuncertainnumber.pba.intervals.methods import subintervalise

    match method:
        case "endpoints":
            vec_itvl = make_vec_interval(vecs)
            return endpoints(vec_itvl, func)
        case "subinterval":
            pass
        case "ga":
            pass
        case "bo":
            pass
        case None:
            return func(vecs)
        case _:
            raise NotImplementedError(f"Method {method} is not supported yet.")


def vec_cartesian_product(*arrays):
    """a vectorised version of the cartesian product"""
    grids = np.meshgrid(*arrays, indexing="ij")
    stacked = np.stack(grids, axis=-1)
    return stacked.reshape(-1, len(arrays))


def i_cartesian_product(a, b):
    """a vectorisation of the interval cartesian product

    todo:
        extend to multiple input arguments
    """
    from pyuncertainnumber import pba

    # Extract bounds
    a_lower = a.lo[:, np.newaxis]  # (2, 1)
    a_upper = a.hi[:, np.newaxis]  # (2, 1)
    b_lower = b.lo[np.newaxis, :]  # (1, 2)
    b_upper = b.hi[np.newaxis, :]  # (1, 2)

    # Broadcast to shape (2, 2)
    cart_lower = np.stack(
        [
            a_lower.repeat(b_lower.shape[1], axis=1),
            np.tile(b_lower, (a_lower.shape[0], 1)),
        ],
        axis=-1,
    )  # shape (2, 2, 2)

    cart_upper = np.stack(
        [
            a_upper.repeat(b_upper.shape[1], axis=1),
            np.tile(b_upper, (a_upper.shape[0], 1)),
        ],
        axis=-1,
    )  # shape (2, 2, 2)

    # Reshape to flat list of interval pairs
    flat_lower = cart_lower.reshape(-1, 2)  # shape (4, 2)
    flat_upper = cart_upper.reshape(-1, 2)  # shape (4, 2)

    return pba.I(lo=flat_lower, hi=flat_upper)


def endpoints(vec_itvl, func):
    """leslie's implementation of endpoints method

    args:
        vec_itvl: a vector type Interval object
        func: the function to be evaluated
    """
    lo, hi = vec_itvl.lo, vec_itvl.hi
    arr = vec_cartesian_product(lo, hi)
    response = func(arr)
    min_response = np.min(response)
    max_response = np.max(response)
    return Interval(min_response, max_response)
