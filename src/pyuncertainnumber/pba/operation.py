# from pyuncertainnumber.pba.intervals.backcalc import additive_bcc

import itertools
from numbers import Number


def convert(un):
    """transform the input un into a Pbox object

    note:
        - theorically 'un' can be {Interval, DempsterShafer, Distribution, float, int}
    """

    from .pbox_base import Pbox
    from .interval import Interval as nInterval
    from .ds import DempsterShafer
    from .distributions import Distribution

    if isinstance(un, nInterval):
        return Pbox(un.left, un.right)
    elif isinstance(un, Pbox):
        return un
    elif isinstance(un, Distribution):
        return un.to_pbox()
    elif isinstance(un, DempsterShafer):
        return un.to_pbox()
    else:
        raise TypeError(f"Unable to convert {type(un)} object to Pbox")


def interval_monte_carlo(f, x, y):
    """
    f: callable
    x, y (Pbox) : Pbox
    """
    from scipy.stats import qmc
    from pyuncertainnumber.pba.intervalOperators import make_vec_interval
    from pyuncertainnumber.pba.aggregation import stacking

    alpha = qmc.LatinHypercube(d=1).random(n=1000)
    x_i = make_vec_interval([x.cuth(p_v) for p_v in alpha])
    y_i = make_vec_interval([y.cuth(p_v) for p_v in alpha])

    container = []
    for _item in itertools.product(x_i, y_i):
        container.append(f(*_item))
    arr_interval = make_vec_interval(container)
    return stacking(arr_interval)


def p_backcalc(a, c, ops):
    """backcal for p-boxes

    args:
        a, c (Pbox):probability box objects
        ops (object) : {'additive_bcc', 'multiplicative_bcc'} whether additive or multiplicative
    """
    from pyuncertainnumber.pba.intervalOperators import make_vec_interval
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
