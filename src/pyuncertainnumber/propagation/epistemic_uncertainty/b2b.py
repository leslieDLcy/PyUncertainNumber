"""leslie's general bound to bound implementation"""

from ...pba.intervals.number import Interval
from ...pba.intervals.intervalOperators import make_vec_interval

import numpy as np


# TODO: integrate GA and BO implementations
# TODO: add discussion of `func` signature (args, collection, matrix) in the notes section
def b2b(
    vecs: Interval | list[Interval],
    func,
    interval_strategy=None,
    style=None,
    n_sub=None,
    **kwargs,
) -> Interval:
    """
    General implementation of a function:

    Y = g(Ix1, Ix2, ..., IxN)

    where Ix1, Ix2, ..., IxN are intervals.

    In a general case, the function g is not necessarily monotonic and g() is a black-box model.
    Optimisation to the rescue and two of them particularly: GA and BO.

    args:
        vecs: a vector Interval or a list or tuple of scalar Interval
        func: performance or response function or a black-box model as in subprocess.
            Expect 2D inputs therefore `func` shall have the matrix signature. See Notes for additional details.
        interval_strategy: the interval_strategy used for interval propagation
            - 'endpoints': only the endpoints
            - 'ga': genetic algorithm
            - 'bo': bayesian optimisation
            - 'diret': direct apply function (the default)
        style: the style only used for subinterval propagation
        **kwargs: additional keyword arguments to be passed to the function

    note:
        'direct' method is not meant to be called directly but to keep as an option during pbox propagation.

    signature:
        This shall be a top-level func as `epistemic_propagation()`.

    returns:
        Interval: the low and upper bound of the response
    """

    vec_itvl = make_vec_interval(vecs)
    match interval_strategy:
        case "endpoints":
            return endpoints(vec_itvl, func)
        case "subinterval":
            return subinterval_method(
                vec_itvl, func, style=style, n_sub=n_sub, **kwargs
            )
        case "ga":
            pass
        case "bo":
            pass
        case "direct":
            return func(vec_itvl)
        case _:
            raise NotImplementedError(
                f"Method {interval_strategy} is not supported yet."
            )


def vec_cartesian_product(*arrays):
    """a vectorised version of the cartesian product

    args:
        arrays: a couple of 1D np.ndarray objects
    """
    grids = np.meshgrid(*arrays, indexing="ij")
    # TODO nice imple but not directrly working with vec Interval object yet
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

    v_np = vec_itvl.to_numpy()
    rows = np.vsplit(v_np, v_np.shape[0])
    arr = vec_cartesian_product(*rows)
    # print(arr.shape)
    # return arr
    response = func(arr)  # func on each row of combination of endpoints
    min_response = np.min(response)
    max_response = np.max(response)
    return Interval(min_response, max_response)


def subinterval_method(vec_itvl, func, style=None, n_sub=None, parallel=False):
    # TODO parallel subinterval
    """leslie's implmentation of subinterval method

    args:
        vec_itvl: a vector type Interval object
        func: the function to be evaluated
        n_sub: number of subintervals
        style: the style used for interval propagation
            - 'direct': direct apply function
            - 'endpoints': only the endpoints
    """
    from pyuncertainnumber.pba.intervals.methods import subintervalise, reconstitute

    if style is None:
        raise ValueError("style must be chosen within {'direct', 'endpoints'}.")
    if n_sub is None:
        raise ValueError("Number of subintervals n_sub must be provided.")

    sub = subintervalise(vec_itvl, n_sub)
    if style == "direct":
        row_n = sub.shape[0]
        return reconstitute([func(sub[IND]) for IND in range(row_n)])
    elif style == "endpoints":
        return reconstitute([endpoints(sub[i], func) for i in range(len(sub))])


"""
POOL:
            except IndexError as e:
                if "too many indices for array" in str(e):
                    print("2D inputs expected but 1D presented:", e)
                    return endpoints(vec_itvl[None, :], func)
                else:
                    raise  # Re-raise if it's a different IndexError
"""
