from .number import Interval
from .methods import log, exp

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
    """additive backcalc"""
    if a.scalar & c.scalar:
        lo = c.lo - a.lo
        hi = c.hi - a.hi
        try:
            return Interval(lo, hi)
        except:
            return Interval(hi, lo)
