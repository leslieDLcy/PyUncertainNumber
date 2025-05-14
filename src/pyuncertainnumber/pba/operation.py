# from pyuncertainnumber.pba.intervals.backcalc import additive_bcc

import itertools
import numpy as np
from numbers import Number
from .params import Params


# there is an new `convert` func
def convert(un):
    """transform the input un into a Pbox object

    note:
        - theorically 'un' can be {Interval, DempsterShafer, Distribution, float, int}
    """

    from .pbox_base import Pbox
    from .dss import DempsterShafer
    from .distributions import Distribution

    if isinstance(un, Pbox):
        return un
    elif isinstance(un, Distribution):
        return un.to_pbox()
    elif isinstance(un, DempsterShafer):
        return un.to_pbox()
    else:
        raise TypeError(f"Unable to convert {type(un)} object to Pbox")


def interval_monte_carlo(f, x, y):
    pass


def p_backcalc(a, c, ops):
    """backcal for p-boxes
    #! incorrect implementation
    args:
        a, c (Pbox):probability box objects
        ops (object) : {'additive_bcc', 'multiplicative_bcc'} whether additive or multiplicative
    """
    from pyuncertainnumber.pba.intervals.intervalOperators import make_vec_interval
    from pyuncertainnumber.pba.aggregation import stacking
    from .pbox_base import Pbox
    from .intervals.number import Interval as I
    from .params import Params

    a_vs = a.to_interval()

    if isinstance(c, Pbox):
        c_vs = c.to_interval()
    elif isinstance(c, Number):
        c_vs = [I(c, c)] * Params.steps

    container = []
    for _item in itertools.product(a_vs, c_vs):
        container.append(ops(*_item))
    # print(len(container))  # shall be 40_000  # checkedout
    arr_interval = make_vec_interval(container)
    return stacking(arr_interval)


def adec(a, c):
    """
    Additive deconvolution: returns b such that a + b ≈ c
    Assumes a, b, c are instances of RandomNbr.

    note:
        implmentation from Scott
    """
    from .intervals.number import Interval as I
    from .pbox_abc import convert_pbox, Staircase

    n = Params.steps
    b = np.zeros(n)  # left bound of B, as in previous b.u[i]
    r = np.zeros(n)
    m = n - 1

    b[0] = c.left[0] - a.left[0]

    for i in range(1, m + 1):
        done = False
        sofar = c.left[i]
        for j in range(i):
            if sofar <= a.left[i - j] + b[j]:
                done = True
        if done:
            b[i] = b[i - 1]
        else:
            b[i] = c.left[i] - a.left[0]

    r[m] = c.right[m] - a.right[m]

    for i in range(m - 1, -1, -1):
        done = False
        sofar = c.right[i]
        for j in range(m, i, -1):
            if sofar >= a.right[i - j + m] + r[j]:
                done = True
        if done:
            r[i] = r[i + 1]
        else:
            r[i] = c.right[i] - a.right[m]

    # Check that bounds do not cross
    bad = any(b[i] > r[i] for i in range(n))

    if bad:
        # Try alternate method
        x = float("inf")
        y = float("-inf")
        for i in range(n):
            y = max(y, c.left[i] - a.left[i])
            x = min(x, c.right[i] - a.right[i])
        B = convert_pbox(I(y, x))
        return B

    # Final bounds check
    for i in range(n):
        if b[i] > r[i]:
            raise ValueError("Math Problem: couldn't deconvolve")
    return Staircase(left=b, right=r)


# * --------------- arithmetic with plain numbers --------------- *#
