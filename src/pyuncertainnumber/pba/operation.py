from pyuncertainnumber.pba.intervals.backcalc import *

import itertools


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


def interval_monte_carlo():
    pass


def p_backcalc(a, c):
    """backcal for p-boxes

    args:
        a, c (Pbox):probability box objects
    """
    from pyuncertainnumber.pba.intervalOperators import make_vec_interval
    from pyuncertainnumber.pba.aggregation import stacking

    # a_vs = a.to_ds().intervals  #backup
    a_vs = a.to_interval()
    c_vs = c.to_interval()
    container = []
    for _item in itertools.product(a_vs, c_vs):
        container.append(backcalc(*_item))
    arr_interval = make_vec_interval(container)
    return stacking(arr_interval)
