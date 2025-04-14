from .number import Interval
from .methods import log, exp
from numbers import Number

"""interval backcalculation"""

# * ------------------ backcalculation ------------------ *#


def backcalc(a, c) -> Interval:
    """backcalculation operation for the Interval object

    signature:
        backcalc(a: Interval, c: Interval) -> Interval
        example: B = backcalc(A, C) # A + B = C
    """
    if a.scalar & c.scalar:
        lo = c.lo - a.lo
        hi = c.hi - a.hi
        return Interval(lo, hi)
    else:
        raise NotImplementedError(
            "backcalculation is not yet implemented for this type of operation"
        )


def factor(a, c):
    """A * B = C

    signature:
        factor(a: Interval, c: Interval) -> Interval
        example: B = factor(A, C) # A * B = C
    """

    if a.scalar & c.scalar:
        return exp(backcalc(log(a), log(c)))
    else:
        raise NotImplementedError(
            "backcalculation is not yet implemented for this type of operation"
        )


# * ------------------ controled backcalculation ------------------ *#


def control_bcc(a, c):
    """controlled backcalculation solution for A + B = C"""
    return -1 * backcalc(c, a)


# * ------------------ mixture ------------------ *#


def additive_bcc(a, c):
    """additive backcalc

    note:
        when c is real number, it is an extreme case of controlled backcalc
        and its results is equivalent to a naive solution
    """
    if isinstance(c, Number):
        c = Interval(c, c)

    if a.scalar & c.scalar:
        lo = c.lo - a.lo
        hi = c.hi - a.hi
        try:
            return Interval(lo, hi)
        except:
            return Interval(hi, lo)


def multiplicative_bcc(a, c):
    """multiplicative backcalc operation to solve B

    note:
        when c is real number, it is an extreme case of controlled backcalc
        and its results is equivalent to a naive solution.
    """
    # TODO shall I call it backcalc or `factor`?

    if isinstance(c, Number):
        c = Interval(c, c)

    if a.scalar & c.scalar:
        lo = c.lo / a.lo
        hi = c.hi / a.hi
        try:
            return Interval(lo, hi)
        except:
            return Interval(hi, lo)
