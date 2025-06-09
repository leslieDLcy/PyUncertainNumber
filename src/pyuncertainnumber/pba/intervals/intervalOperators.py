from functools import singledispatch
import numpy as np
from . import intervalise, Interval
from ...nlp.language_parsing import parse_interval_expression, hedge_interpret
from numbers import Number

""" operations for generic Interval objects """

# see the hedged interpretation for Interval in `nlp/language_parsing.py`


def parse_bounds(b):
    """top-level function that universally parses scalar and vector bounds"""
    try:
        return wc_scalar_interval(b)
    except Exception:
        return make_vec_interval(b)


# * ---------------------make scalar interval object --------------------- *#


def wc_scalar_interval_feature(*args):
    """wildcard scalar interval

        This function is used to parse a scalar bound into an Interval object.
        It can handle various input types such as lists, tuples, and strings.
        If the input is a string, it attempts to interpret it using the
        `hedge_interpret` function or parse it as an interval expression.
        If the input is a single number, it creates an Interval with that number
        as both bounds.


    note:
        This function is a beta version of `wc_scalar_interval` which is meant to test the API signature.
        If run into error, then resort back to `wc_scalar_interval`.
    """
    from ...characterisation.uncertainNumber import UncertainNumber as UN

    if len(args) == 1:
        bound = args[0]
    elif len(args) == 2:
        return Interval(*args)
    else:
        raise ValueError("wc_scalar_interval only accepts 1 or 2 arguments")

    if isinstance(bound, list):
        return Interval(*bound)
    elif isinstance(bound, tuple):
        return Interval(*bound)
    elif isinstance(bound, Interval):
        return bound
    elif isinstance(bound, Number):
        return Interval(bound, bound)
    elif isinstance(bound, str):
        try:
            return hedge_interpret(bound)
        except Exception:
            pass
        try:
            return parse_interval_expression(bound)
        except Exception:
            raise ValueError("Invalid input")
    elif isinstance(bound, UN):
        return bound.construct
    else:
        raise TypeError("Unsupported type for interval creation")


def wc_scalar_interval(bound):
    """wildcard scalar interval

    This function is used to parse a scalar bound into an Interval object.
    It can handle various input types such as lists, tuples, and strings.
    If the input is a string, it attempts to interpret it using the
    `hedge_interpret` function or parse it as an interval expression.
    If the input is a single number, it creates an Interval with that number
    as both bounds.
    """
    from ...characterisation.uncertainNumber import UncertainNumber as UN

    if isinstance(bound, list):
        return Interval(*bound)
    elif isinstance(bound, tuple):
        return Interval(*bound)
    elif isinstance(bound, Interval):
        return bound
    elif isinstance(bound, Number):
        return Interval(bound, bound)
    elif isinstance(bound, str):
        try:
            return hedge_interpret(bound)
        except Exception:
            pass
        try:
            return parse_interval_expression(bound)
        except Exception:
            raise ValueError("Invalid input")
    elif isinstance(bound, UN):
        return bound.construct
    else:
        raise TypeError("Unsupported type for interval creation")


""" old implementation

note: it is deprecated due to it cannot handle custom Class e.g. UN 
because it has to import UN which creates circular import issue.
"""


# @singledispatch
# def wc_scalar_interval(bound):
#     """wildcard scalar interval"""
#     return Interval(bound)


# @wc_scalar_interval.register(list)
# def _list(bound: list):
#     return Interval(*bound)


# @wc_scalar_interval.register(tuple)
# def _tuple(bound: tuple):
#     return Interval(*bound)


# @wc_scalar_interval.register(Interval)
# def _marco_interval_like(bound: Interval):
#     return bound


# @wc_scalar_interval.register(Number)
# def _scalar(bound: Number):
#     return Interval(bound, bound)


# @wc_scalar_interval.register(str)
# def _scalar(bound: str):

#     try:
#         return hedge_interpret(bound)
#     except Exception:
#         pass

#     try:
#         return parse_interval_expression(bound)
#     except Exception:
#         raise ValueError("Invalid input")


# * ---------------------make vector interval object --------------------- *#


def make_vec_interval(vec):
    """parse into a vector interval"""
    from ...characterisation.uncertainNumber import UncertainNumber as UN

    assert len(vec) > 1, "Interval must have more than one element"

    if isinstance(vec, Interval):
        return vec
    elif isinstance(vec[0], Interval | list | tuple | np.ndarray):
        try:
            return intervalise(vec)
        except Exception:
            return intervalise(list(vec))
    elif isinstance(vec[0], UN):
        return intervalise([v.construct for v in vec])
    else:
        raise NotImplementedError


# * ---------------------mean func --------------------- *#


@singledispatch
def mean(x):
    return np.mean(x)


@mean.register(np.ndarray)
def _arraylike(x):
    return np.mean(x)


@mean.register(Interval)
def _intervallike(x):
    return sum(x) / len(x)
