"""leslie's general bound to bound implementation"""

from ...pba.intervals.number import Interval


def b2b(xs, func, *args, **kwargs) -> Interval:
    """
    General implementation of a function:

    Y = g(Ix1, Ix2, ..., IxN)

    where Ix1, Ix2, ..., IxN are intervals.

    In a general case, the function g is not necessarily monotonic and g() is a black-box model.
    Optimisation to the rescue and two of them particularly: GA and BO.

    args:
        xs: list of intervals
        func: performance or response function
        *args: additional arguments to be passed to the function
        **kwargs: additional keyword arguments to be passed to the function

    returns:
        Interval: the low and upper bound of the response
    """
    pass
