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
    match method:
        case "endpoints":
            vec_itvl = make_vec_interval(vecs)
            lo, hi = vec_itvl.lo, vec_itvl.hi
            arr = vec_cartesian_product(lo, hi)
            response = func(arr)
            min_response = np.min(response)
            max_response = np.max(response)
            return Interval(min_response, max_response)
        case "ga":
            pass
        case "bo":
            pass
        case _:
            raise NotImplementedError(f"Method {method} is not supported yet.")


def vec_cartesian_product(*arrays):
    """a vectorised version of the cartesian product"""
    grids = np.meshgrid(*arrays, indexing="ij")
    stacked = np.stack(grids, axis=-1)
    return stacked.reshape(-1, len(arrays))
