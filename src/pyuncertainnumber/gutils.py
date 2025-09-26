from .characterisation.uncertainNumber import UncertainNumber
from pyuncertainnumber import Interval, Pbox, DempsterShafer
from .pba.distributions import Distribution
from numbers import Number

# * ---------------------helper functions  --------------------- *#


def pass_down_units(a, b, ops, t):
    """pass down the unit of the uncertain number

    args:
        - a: the first uncertain number
        - b: the second uncertain number
        - ops: the operation to be performed
        - t: the result uncertain number of the operation
    """
    if is_un(b) == 0:
        try:
            new_q = ops(a._physical_quantity, b * a._physical_quantity.units)
        except Exception:
            new_q = ops(a._physical_quantity, b)
    elif is_un(b) == 1:
        new_q = ops(a._physical_quantity, b._physical_quantity)

    t.physical_quantity = new_q
    return t


def is_un(sth):
    """utility function to decide the essence of the object

    returns:
        - 0: if sth is a regular number, float or int
        - 1: if sth is an UncertainNumber object
        - 2: if sth is a construct in {Interval, Pbox, DempsterShafer, or Distribution}
    """

    if isinstance(sth, Number):
        return 0
    elif isinstance(sth, UncertainNumber):
        return 1
    elif isinstance(sth, Interval | Pbox | DempsterShafer | Distribution):
        return 2


def exist_un(a_list) -> bool:
    """check if there is any UN object in the list"""
    return any(is_un(x) == 1 for x in a_list)
