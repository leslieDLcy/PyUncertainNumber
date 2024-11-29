from functools import singledispatch
from intervals import Interval
import numpy as np

# * ---------------------mean func --------------------- *#


@singledispatch
def mean(x):
    return np.mean(x)


@mean.register(np.ndarray)
def _arraylike(x):
    return np.mean(x)


@mean.register(Interval)
def _intervallike(x):
    return interval_mean_func(x)

# * ---------------------std func --------------------- *#


def std():
    pass


# * ---------------------var func --------------------- *#


def var():
    pass


# * ---------------------round func --------------------- *#
def round():
    pass
