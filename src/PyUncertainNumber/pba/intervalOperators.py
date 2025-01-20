from functools import singledispatch
import numpy as np
from .interval import Interval as nInterval
from .intervals import intervalise, Interval
from ..nlp.language_parsing import parse_interval_expression, hedge_interpret

""" operations for generic Interval objects """

# see the hedged interpretation for Interval in `nlp/language_parsing.py`


@singledispatch
def parse_bounds(bounds):
    """parse the self.bounds argument"""
    return wc_interval(bounds)


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
def wc_interval(bound):
    """wildcard scalar interval"""
    return nInterval(bound)


@wc_interval.register(list)
def _arraylike(bound: list):
    return nInterval(bound)


@wc_interval.register(Interval)
def _marco_interval_like(bound: Interval):
    return nInterval(np.ndarray.item(bound.lo), np.ndarray.item(bound.hi))


@wc_interval.register(nInterval)
def _nick_interval_like(bound: nInterval):
    return bound


# * ---------------------make vector interval object --------------------- *#


def make_vec_interval(vec):
    """vector interval implementation tmp"""
    assert len(vec) > 1, "Interval must have more than one element"

    if isinstance(vec, Interval):
        return vec

    elif isinstance(vec[0], nInterval):
        lo_endpoints = [un.left for un in vec]
        hi_endpoints = [un.right for un in vec]
        return Interval(lo_endpoints, hi_endpoints)

    elif isinstance(vec[0], list | tuple | np.ndarray):
        # lo_endpoints = [un[0] for un in vec]
        # hi_endpoints = [un[1] for un in vec]
        # return Interval(lo_endpoints, hi_endpoints)
        return intervalise(vec)
    else:
        print("not implemented yet")

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


# * ---------------------std func --------------------- *#


def std():
    pass


# * ---------------------var func --------------------- *#


def var():
    pass


# * ---------------------round func --------------------- *#
def roundInt():
    """outward rounding to integer"""
    pass
