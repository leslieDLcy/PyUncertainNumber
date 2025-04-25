from functools import singledispatch
import numpy as np
from .intervals import intervalise, Interval
from ..nlp.language_parsing import parse_interval_expression, hedge_interpret
from numbers import Number

""" operations for generic Interval objects """

# see the hedged interpretation for Interval in `nlp/language_parsing.py`


@singledispatch
def parse_bounds(bounds):
    """parse the self.bounds argument"""
    return wc_scalar_interval(bounds)


@parse_bounds.register(str)
def _str(bounds: str):

    try:
        return hedge_interpret(bounds)
    except Exception:
        pass

    try:
        return parse_interval_expression(bounds)
    except Exception:
        raise ValueError("Invalid input")


# * ---------------------make scalar interval object --------------------- *#


@singledispatch
def wc_scalar_interval(bound):
    """wildcard scalar interval"""
    return Interval(bound)


@wc_scalar_interval.register(list)
def _list(bound: list):
    return Interval(*bound)


@wc_scalar_interval.register(tuple)
def _tuple(bound: tuple):
    return Interval(*bound)


@wc_scalar_interval.register(Interval)
def _marco_interval_like(bound: Interval):
    return bound


@wc_scalar_interval.register(Number)
def _scalar(bound: Number):
    return Interval(bound, bound)


# * ---------------------make vector interval object --------------------- *#


def make_vec_interval(vec):
    """transform the argument into a vector interval tmp"""

    assert len(vec) > 1, "Interval must have more than one element"

    if isinstance(vec, Interval):
        return vec
    elif isinstance(vec[0], Interval | list | tuple | np.ndarray):
        return intervalise(vec)
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
