import numpy
from numpy.random import rand, randint

from .number import Interval
from .methods import intervalise

LEFT_BOUND = -1_000
RIGHT_BOUND = 1_000


def uniform_endpoints(
    n: int = 2,
    left_bound: float = None,
    right_bound: float = None,
    kind: type = float,
    shape: tuple = None,
):  # when N=2 generates two intervals
    """
    Draws endpoints from a uniform distribution.
    It was created as an interval random generator.
    """
    if left_bound is None:
        left_bound = LEFT_BOUND
    if right_bound is None:
        right_bound = RIGHT_BOUND
    if shape is None:
        if n == 1:
            if kind == float:
                improper = left_bound + rand(
                    2,
                ) * (right_bound - left_bound)
            elif kind == int:
                improper = randint(left_bound, high=right_bound, size=(2,))
            else:
                return NotImplemented
            if improper[0] <= improper[1]:
                return Interval(lo=improper[0], hi=improper[1])
            else:
                return Interval(lo=improper[1], hi=improper[0])
        else:
            if kind == float:
                improper = left_bound + rand(n, 2) * (right_bound - left_bound)
            elif kind == int:
                improper = randint(left_bound, high=right_bound, size=(n, 2))
            else:
                return NotImplemented
            improper_lo, improper_hi = improper[:, 0], improper[:, 1]
    else:
        if kind == float:
            improper_lo = left_bound + rand(*shape) * (right_bound - left_bound)
            improper_hi = left_bound + rand(*shape) * (right_bound - left_bound)
        elif kind == int:
            improper_lo = randint(left_bound, high=right_bound, size=shape)
            improper_hi = randint(left_bound, high=right_bound, size=shape)
        else:
            return NotImplemented
    swap = improper_lo >= improper_hi
    proper_lo, proper_hi = improper_lo.copy(), improper_hi.copy()
    if numpy.sum(swap) > 0:
        proper_lo[swap], proper_hi[swap] = proper_hi[swap], proper_lo[swap]
    return Interval(lo=proper_lo, hi=proper_hi)


def create_two_large_interval_matrices(shape, left_bound=0, right_bound=1):
    # shape=(100000,100000)
    x = uniform_endpoints(shape=shape, left_bound=left_bound, right_bound=right_bound)
    y = uniform_endpoints(shape=shape, left_bound=left_bound, right_bound=right_bound)
    return x, y
