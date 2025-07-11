from __future__ import annotations
from typing import TYPE_CHECKING

"""leslie's general bound to bound implementation"""

from ...pba.intervals.number import Interval
from ...pba.intervals.intervalOperators import make_vec_interval
import numpy as np

if TYPE_CHECKING:
    from ...pba.intervals import Interval


# TODO: integrate GA and BO implementations
# TODO: add discussion of `func` signature (args, collection, matrix) in the notes section
def b2b(
    vars: Interval | list[Interval],
    func: callable,
    interval_strategy: str = None,
    style: str = None,
    n_sub: int = None,
    **kwargs,
) -> Interval:
    """General implementation of interval propagation through a function:

    .. math::
        Y = g(I_{x1}, I_{x2}, ..., I_{xn})

    where :math:`I_{x1}, I_{x2}, ..., I_{xn}` are intervals.

    In a general case, the function :math:`g` is not necessarily monotonic and :math:`g` may be a black-box model.
    Optimisation to the rescue and two of them particularly: GA and BO.

    args:
        vars (Interval): a vector Interval or a list or tuple of scalar Interval

        func (callable): performance or response function or a black-box model as in subprocess.
            Expect 2D inputs therefore `func` shall have the matrix signature. See Notes for additional details.

        interval_strategy (str): the interval_strategy used for interval propagation. The choice shall be compatible with the response function signature. Seethe notes below for additional details.

            - 'endpoints': only the endpoints

            - 'ga': genetic algorithm

            - 'bo': bayesian optimisation

            - 'diret': direct apply function to the input intervals (the default)

        style (str):
            the style only used for subinterval propagation, including {''direct'', ''endpoints''}.

        **kwargs: additional keyword arguments to be passed to the function


    tip:
        This serves as a top-level for generic interval propagation .

    caution:
        ``interval_strategy`` suggests the method of interval propagation (e.g. 'direct' or 'endpoints', or 'ga', or 'bo'),  whhile ``style`` is only used for subinterval propagation.
        This is to say that what strategy (whether 'direct' or 'endpoints') will be chosen for the sub-intervals.

    danger:
        There are some subtleties about the calling signature of the propagating function. For ``endpoints`` strategy/style, the function `func` should have a vectorised signature
        as it is expecting a 2D numpy array, whereas for the ``direct`` strategy/style, it is expecting to take individual scalar inputs.
        It is recommended to write a function which implements both signature, as seen in the example below.

    returns:
        Interval: the low and upper bound of the response


    example:
        >>> from pyuncertainnumber import b2b
        >>> import numpy as np
        >>> import pyuncertainnumber as pba

        >>> # Define a universal function that handles both vectorised and iterable inputs
        >>> def bar_universal(x):
        ...     if isinstance(x, np.ndarray):
        ...         if x.ndim == 1:
        ...             x = x[None, :]
        ...         return x[:, 0] ** 3 + x[:, 1] + 5  # vectorised signature
        ...     else:
        ...         return x[0] ** 3 + x[1] + 5  # iterable signature

        >>> # Define input intervals
        >>> a = pba.I(3., 5.)
        >>> b = pba.I(6., 26.)

        >>> # using the {'endpoints', "direct", ga", "bo"} strategy
        >>> b2b(vars=[a, b],
        ...     func=bar_universal,
        ...     interval_strategy='endpoints')  # replace with {"direct", ga", "bo"}
        [38.0, 156.0]

        >>> # using the 'subinterval' strategy
        >>> b2b(vars=[a, b],
        ...     func=bar_universal,
        ...     interval_strategy='subinterval',
        ...     style='endpoints',
        ...     n_sub=2)
        [38.0, 156.0]

        >>> # in comparison, one can compare with the result of interval arithmetic
        >>> def bar_individual(x1, x2):
        ...     return x1 ** 3 + x2 + 5  # individual signature
        >>> bar_individual(a, b)
        [38.0, 156.0]
    """

    vec_itvl = make_vec_interval(vars)
    match interval_strategy:
        case "endpoints":
            return endpoints(vec_itvl, func)
        case "subinterval":
            return subinterval_method(
                vec_itvl, func, style=style, n_sub=n_sub, **kwargs
            )
        case "ga":
            from ...opt.get_range import get_range_GA
            from pyuncertainnumber import EpistemicDomain

            ep = EpistemicDomain(vec_itvl)
            opt_result = get_range_GA(
                f=func,
                dimension=len(vec_itvl),
                varbound=ep.to_GA_varBounds(),
                verbose=False,
            )
            return opt_result[0]  # return the interval only
        case "bo":
            from ...opt.get_range import get_range_BO
            from pyuncertainnumber import EpistemicDomain

            ep = EpistemicDomain(vec_itvl)

            opt_result = get_range_BO(
                f=func,
                dimension=len(vec_itvl),
                xc_bounds=ep.to_BayesOptBounds(),
                verbose=False,
            )
            return opt_result[0]  # return the interval only
        case "direct":
            return func(vec_itvl)
        case _:
            raise NotImplementedError(
                f"Method {interval_strategy} is not supported yet."
            )


def endpoints(vec_itvl: Interval, func) -> Interval:
    """Implementation of endpoints method

    args:
        vec_itvl (Interval): a vector type Interval object
        func (callable): the function to be evaluated. See notes about function signature.

    note:
        The function `func` is expected to accept a 2D numpy array of shape :math:`(2^d, d)` where `d` is the dimension of the vector interval.
        Therefore, the function  should have a vectorised signature, as opposted to taking individual scalar inputs.


    Example:
        >>> from pyuncertainnumber import pba
        >>> v = pba.I([1, 2], [3, 4])  # a vector interval with two dimensions
        >>> def bar(x): return x[:, 0] ** 3 + x[:, 1] + 5
        >>> endpoints(v, bar)
        Interval(8.0, 36.0)
    """

    v_np = vec_itvl.to_numpy()
    rows = np.vsplit(v_np, v_np.shape[0])
    arr = vec_cartesian_product(*rows)
    # print(arr.shape)  # array of shape (2**n, 2)

    response = func(arr)  # func on each row of combination of endpoints
    min_response = np.min(response)
    max_response = np.max(response)
    return Interval(min_response, max_response)


def subinterval_method(
    vec_itvl: Interval, func, style=None, n_sub=None, parallel=False
) -> Interval:
    # TODO parallel subinterval
    """Implmentation of subinterval method which splits subintervals and reconstitutes later.

    Args:
        vec_itvl (Interval): a vector type Interval object

        func (callable): the function to be evaluated. See notes about function signature.

        n_sub (int): number of subintervals

        style (str): the style used for interval propagation which shall be compatible with the response function signature.

            - 'direct': direct apply function

            - 'endpoints': only the endpoints are propagated

    danger:
        There are some subtleties about function signature. For ``endpoints`` style, the function `func` should have a vectorised signature
        as it is expecting a 2D numpy array, whereas for the `direct` style, it is expecting to take individual scalar inputs

    Example:
        >>> from pyuncertainnumber import pba
        >>> v = pba.I([1, 2], [3, 4])
        >>> def bar(x): return x[0] ** 3 + x[1] + 5
        >>> b2b.subinterval_method(vec_itvl=v, func=bar, style='direct', n_sub=2)
        >>> def bar_vec(x): return x[:, 0] ** 3 + x[:, 1] + 5
        >>> b2b.subinterval_method(vec_itvl=v, func=bar_vec, style='endpoints', n_sub=2)
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


"""
POOL:
            except IndexError as e:
                if "too many indices for array" in str(e):
                    print("2D inputs expected but 1D presented:", e)
                    return endpoints(vec_itvl[None, :], func)
                else:
                    raise  # Re-raise if it's a different IndexError
"""
