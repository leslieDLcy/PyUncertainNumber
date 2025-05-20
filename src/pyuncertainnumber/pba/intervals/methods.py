"""
------------------------------
cre: Feb 2022

web: github.com/marcodeangelis
org: Univerity of Liverpool

MIT License
------------------------------

These methods are designed to behave neutrally on non-interval inputs.
So, if a non-interval is passed equivalent rules for floats apply.

Interval to float methods, IR -> R:

Interval to bool methods, IR -> {0,1}:

Binary operations, IR2 -> IR
Unary operations, IR -> IR

Parser, R^(nx2) -> IR^n, R^(mxnx2) -> IR^(mxn), R^(2xmxn) -> IR^(mxn)
This method turns an array of compatible dimension into an interval (array).

Subintervalisation methods, IR -> IR^n.

"""

from __future__ import annotations
from typing import Sequence, Sized, Iterable, Optional, Any, Tuple, Union
from itertools import product
import numpy
from numpy import ndarray, asarray, vstack, linspace, zeros, argmax
from .number import Interval, MACHINE_EPS, lo, hi, width, mid

numpy_min = numpy.min
numpy_max = numpy.max
numpy_sqrt = numpy.sqrt
numpy_abs = numpy.abs
numpy_exp = numpy.exp
numpy_sum = numpy.sum
numpy_sin = numpy.sin
numpy_cos = numpy.cos
numpy_tan = numpy.tan
# numpy_cot = numpy.cotang
numpy_pi = numpy.pi
numpy_inf = numpy.inf
numpy_transpose = numpy.transpose


#####################################################################################
# unary.py
#####################################################################################
# Interval to interval methods. Unary.


def abs(x: Interval):
    """
    Return the absolute value of an Interval.

    If x is not of class Interval, absolute value is returned assuming input is numerical.

    If x is neither a number (neither Interval not numeric), numpy will throw an exception.

    """
    x_lo_abs = numpy.abs(lo(x))
    if is_Interval(x):
        zero_in_x = contain(x, 0)
        x_hi_abs = numpy.abs(hi(x))
        a = numpy.min((x_lo_abs, x_hi_abs), axis=0)
        if x.unsized:
            if zero_in_x:
                a = 0
        else:
            a[zero_in_x] = 0
        b = numpy.max((x_lo_abs, x_hi_abs), axis=0)
        return Interval(a, b)
    return x_lo_abs


def sqrt(x: Interval):
    """
    Return the square root of an Interval.

    If x is not of class Interval, the square root is returned assuming input is numerical.

    If x is neither a number (neither Interval not numeric), numpy will throw an exception.

    """
    x_lo_sqrt = numpy.sqrt(lo(x))
    if is_Interval(x):
        x_hi_sqrt = numpy.sqrt(hi(x))
        return Interval(x_lo_sqrt, x_hi_sqrt)
    return x_lo_sqrt


def exp(x: Interval):
    if is_not_Interval(x):
        return numpy_exp(x)
    return Interval(numpy_exp(lo(x)), numpy_exp(hi(x)))


def log(x: Interval):
    if is_not_Interval(x):
        return numpy.log(x)
    else:
        assert x.lo > 0, "interval has to be positive"
    return Interval(numpy.log(lo(x)), numpy.log(hi(x)))


#####################################################################################
# binary.py
#####################################################################################
# Binary methods between two intervals
# 2-interval to interval. Bianry.


def max(x: Interval, y: Interval):
    if all([is_not_Interval(x), is_not_Interval(y)]):
        return numpy.max((x, y), axis=0)
    a = numpy.max((lo(x), lo(y)), axis=0)
    b = numpy.max((hi(x), hi(y)), axis=0)
    return Interval(a, b)


def min(x: Interval, y: Interval):
    if all([is_not_Interval(x), is_not_Interval(y)]):
        return numpy.min((x, y), axis=0)
    a = numpy.min((lo(x), lo(y)), axis=0)
    b = numpy.min((hi(x), hi(y)), axis=0)
    return Interval(a, b)


#####################################################################################
# trig.py
#####################################################################################


def sin(x: Interval):
    """
    Implementation of Interval Arithmetic in CORA 2016

    Matthias Althoff and Dmitry Grebenyuk

    EPiC Series in Computing Volume 43, 2017, Pages 91-105

    ARCH16. 3rd International Workshop on Applied Verification for Continuous and Hybrid Systems
    """
    if not (is_Interval(x)):
        return numpy_sin(x)  # int, float, ndarray

    if not (x.scalar):
        return sin_vector(x)

    twopi = 2 * numpy_pi
    pihalf = numpy_pi / 2

    if width(x) >= twopi:
        return Interval(-1, 1)

    domain1 = Interval(0, pihalf)
    domain2 = Interval(pihalf, 3 * pihalf)
    domain3 = Interval(3 * pihalf, twopi)

    yl = x.lo % twopi
    yh = x.hi % twopi
    y = Interval(lo=yl, hi=yh)

    sin_l = numpy_sin(yl)
    sin_h = numpy_sin(yh)

    if contain(domain1, y) & (yl <= yh):
        return Interval(sin_l, sin_h)
    if contain(domain2, y) & (yl <= yh):
        return Interval(sin_h, sin_l)
    if contain(domain3, y) & (yl <= yh):
        return Interval(sin_l, sin_h)

    case1a = contain(domain1, yl) & contain(domain1, yh) & (yl > yh)
    case1b = contain(domain1, yl) & contain(domain3, yh)
    case1c = contain(domain2, yl) & contain(domain2, yh) & (yl > yh)
    case1d = contain(domain3, yl) & contain(domain3, yh) & (yl > yh)

    case2a = contain(domain1, yl) & contain(domain1, yh) & (yl <= yh)
    case2b = contain(domain3, yl) & contain(domain1, yh)
    case2c = contain(domain3, yl) & contain(domain3, yh) & (yl <= yh)

    case3a = contain(domain1, yl) & contain(domain2, yh)
    case3b = contain(domain3, yl) & contain(domain2, yh)

    case4a = contain(domain2, yl) & contain(domain1, yh)
    case4b = contain(domain2, yl) & contain(domain3, yh)

    case5 = contain(domain2, yl) & contain(domain2, yh) & (yl <= yh)

    if case1a | case1b | case1c | case1d:
        return Interval(-1, 1)
    if case2a | case2b | case2c:
        return Interval(sin_l, sin_h)
    if case3a | case3b:
        return Interval(min(sin_l, sin_h), 1)
    if case4a | case4b:
        return Interval(-1, max(sin_l, sin_h))
    if case5:
        return Interval(sin_h, sin_l)

    pass


def sin_vector(x: Interval):  # vectorised version of sin().

    if x.unsized:
        return sin(x)

    twopi = 2 * numpy_pi
    pihalf = numpy_pi / 2

    mask1a = width(x) >= twopi

    domain1 = Interval(0, pihalf)
    domain2 = Interval(pihalf, 3 * pihalf)
    domain3 = Interval(3 * pihalf, twopi)

    yl = x.lo % twopi
    yh = x.hi % twopi
    y = Interval(yl, yh)

    sin_l = numpy_sin(yl)
    sin_h = numpy_sin(yh)

    # [l,h] all else
    a = sin_l.copy()
    b = sin_h.copy()

    # [-1,1]
    mask3a = contain(domain1, yl) & contain(domain1, yh) & (yl > yh)
    mask3b = contain(domain1, yl) & contain(domain3, yh)
    mask3c = contain(domain2, yl) & contain(domain2, yh) & (yl > yh)
    mask3d = contain(domain3, yl) & contain(domain3, yh) & (yl > yh)
    case1 = mask1a | mask3a | mask3b | mask3c | mask3d
    a[case1] = -1
    b[case1] = 1
    if all(case1):
        return Interval(lo=a, hi=b)
    # [h,l]
    mask2b = (
        contain(domain2, yl[~case1])
        & contain(domain2, yh[~case1])
        & (yl[~case1] <= yh[~case1])
    )  # return Interval(sin_h,sin_l)
    case2 = mask2b
    a[case2] = sin_h[case2]
    b[case2] = sin_l[case2]
    # [min, 1]
    mask5a = contain(domain1, yl[~case1]) & contain(domain2, yh[~case1])
    mask5b = contain(domain3, yl[~case1]) & contain(domain2, yh[~case1])
    case3 = mask5a | mask5b
    a[case3] = min(sin_l[case3], sin_h[case3])
    b[case3] = 1
    # [-1, max]
    mask6a = contain(domain2, yl[~case1]) & contain(domain1, yh[~case1])
    mask6b = contain(domain2, yl[~case1]) & contain(domain3, yh[~case1])
    case4 = mask6a | mask6b
    a[case4] = -1
    b[case4] = max(sin_l[case4], sin_h[case4])
    return Interval(lo=a, hi=b)


def cos(x: Interval):
    """
    Implementation of Interval Arithmetic in CORA 2016

    Matthias Althoff and Dmitry Grebenyuk

    EPiC Series in Computing Volume 43, 2017, Pages 91-105

    ARCH16. 3rd International Workshop on Applied Verification for Continuous and Hybrid Systems
    """
    if not (is_Interval(x)):
        return numpy_cos(x)  # int, float, ndarray

    if not (x.scalar):
        return cos_vector(x)

    twopi = 2 * numpy_pi

    # [-1,1] aka case 0
    if width(x) >= twopi:
        return Interval(-1, 1)

    domain1 = Interval(0, numpy_pi)
    domain2 = Interval(numpy_pi, 2 * numpy_pi)

    yl = x.lo % twopi
    yh = x.hi % twopi
    y = Interval(lo=yl, hi=yh)

    cos_l = numpy_cos(yl)
    cos_h = numpy_cos(yh)

    # [-1,1]
    case1a = (yh < yl) & contain(domain1, yl) & contain(domain1, yh)
    case1b = (yh < yl) & contain(domain2, yl) & contain(domain2, yh)
    # [cos_l, cos_h]
    case2a = (yl <= yh) & contain(domain2, yl) & contain(domain2, yh)
    # [min(cos_l, cos_h), 1]
    case3a = contain(domain2, yl) & contain(domain1, yh)
    # [-1, max(cos_l, cos_h)]
    case4a = contain(domain1, yl) & contain(domain2, yh)
    # [cos_h, cos_l]
    case5a = (yl <= yh) & contain(domain1, yl) & contain(domain1, yh)

    if case1a | case1b:
        return Interval(-1, 1)
    if case2a:
        return Interval(cos_l, cos_h)
    if case3a:
        return Interval(min(cos_l, cos_h), 1)
    if case4a:
        return Interval(-1, max(cos_l, cos_h))
    if case5a:
        return Interval(cos_h, cos_l)


def cos_vector(x: Interval):  # vectorised version of cos()
    if x.unsized:
        return sin(x)

    twopi = 2 * numpy_pi

    case0 = width(x) >= twopi

    domain1 = Interval(0, numpy_pi)
    domain2 = Interval(numpy_pi, 2 * numpy_pi)

    yl = x.lo % twopi
    yh = x.hi % twopi

    cos_l = numpy_cos(yl)
    cos_h = numpy_cos(yh)

    a = cos_l.copy()
    b = cos_h.copy()

    # [-1,1]
    case1a = (yh < yl) & contain(domain1, yl) & contain(domain1, yh)
    case1b = (yh < yl) & contain(domain2, yl) & contain(domain2, yh)
    case1 = case0 | case1a | case1b
    a[case1] = -1
    b[case1] = 1
    # [cos_l, cos_h]
    # case2 = (yl<=yh) & contain(domain2,yl) & contain(domain2,yh)
    # a[case2] = cos_l[case2]
    # b[case2] = cos_h[case2]
    # [min(cos_l, cos_h), 1]
    case3 = contain(domain2, yl) & contain(domain1, yh)
    a[case3] = min(cos_l[case3], cos_h[case3])
    b[case3] = 1
    # [-1, max(cos_l, cos_h)]
    case4 = contain(domain1, yl) & contain(domain2, yh)
    a[case4] = -1
    b[case4] = max(cos_l[case4], cos_h[case4])
    # [cos_h, cos_l]
    case5 = (yl <= yh) & contain(domain1, yl) & contain(domain1, yh)
    a[case5] = cos_h[case5]
    b[case5] = cos_l[case5]
    return Interval(lo=a, hi=b)


def tan(x: Interval):
    """
    Implementation of Interval Arithmetic in CORA 2016

    Matthias Althoff and Dmitry Grebenyuk

    EPiC Series in Computing Volume 43, 2017, Pages 91-105

    ARCH16. 3rd International Workshop on Applied Verification for Continuous and Hybrid Systems
    """

    if not (is_Interval(x)):
        return numpy_tan(x)  # int, float, ndarray

    if not (x.scalar):
        return tan_vector(x)

    pihalf = numpy_pi / 2

    domain1 = Interval(0, pihalf)
    domain2 = Interval(pihalf, numpy_pi)

    zl = x.lo % numpy_pi
    zh = x.hi % numpy_pi

    # [-∞, ∞]
    case1a = width(x) > numpy_pi
    case1b = (zh < zl) & contain(domain1, zl) & contain(domain1, zh)
    case1c = (zh < zl) & contain(domain2, zl) & contain(domain2, zh)
    case1d = contain(domain1, zl) & contain(domain2, zh)

    # [tan_l, tan_h]
    case2a = (zl <= zh) & contain(domain1, zl) & contain(domain1, zh)
    case2b = (zl <= zh) & contain(domain2, zl) & contain(domain2, zh)

    if case1a | case1b | case1c | case1d:
        return Interval(-numpy_inf, numpy_inf)
    if case2a | case2b:
        return Interval(tan(zl), tan(zh))
    else:
        return Interval(tan(zl), tan(zh))


def tan_vector(x: Interval):  # Vectorised version of tan().
    if x.unsized:
        return tan(x)

    pihalf = numpy_pi / 2

    domain1 = Interval(0, pihalf)
    domain2 = Interval(pihalf, numpy_pi)

    zl = x.lo % numpy_pi
    zh = x.hi % numpy_pi

    tan_l = tan(zl)
    tan_h = tan(zh)

    a = tan_l.copy()
    b = tan_h.copy()

    # [-∞, ∞]
    case1a = width(x) > numpy_pi
    case1b = (zh < zl) & contain(domain1, zl) & contain(domain1, zh)
    case1c = (zh < zl) & contain(domain2, zl) & contain(domain2, zh)
    case1d = contain(domain1, zl) & contain(domain2, zh)
    case1 = case1a | case1b | case1c | case1d
    a[case1] = -numpy_inf
    b[case1] = numpy_inf

    # #[tan_l, tan_h]
    # case2a = (zl<=zh) & contain(domain1,zl) & contain(domain1,zh)
    # case2b = (zl<=zh) & contain(domain2,zl) & contain(domain2,zh)
    return Interval(lo=a, hi=b)


# Interval to ndarray[float]
# def linspace():


# def union(): pass # spell the difference between union and subpaving.
# def intersection():
# def difference():
# def set_difference(x:Interval,y:Interval):


#####################################################################################
# set.py
#####################################################################################
# Interval to bool methods, Binary.
# ...
def straddle_zero(x: Interval) -> bool:
    if x.unsized:
        return (lo(x) <= 0) & (hi(x) >= 0)
    else:
        return any((lo(x).flatten() <= 0) & (hi(x).flatten() >= 0))


def intersect(x: Interval, y: Interval):
    return ~((x < y) | (y < x))  # commutative


def contain(x: Interval, y: Interval):
    return (lo(x) <= lo(y)) & (hi(x) >= hi(y))  # x contain y


def almost_contain(x: Interval, y: Interval, tol=1e-9):
    return (
        lo(y) -
        # x contain y
        lo(x)
        > -tol
    ) & (hi(x) - hi(y) > -tol)


def intersect_vector(x_: Interval, y_: Interval):
    """
    This function checks if the focal elements x, intersect the subpaving y.

    x: A n-list of d-boxes or d-intervals, e.g. a subpaving. x.shape=(r,d)
    y: A m-list of d-boxes or d-intervals, e.g. a focal element. y.shape=(p,d)

    out: A (rp)-list of d-arrays of booleans
    """
    x = intervalise(x_)
    y = intervalise(y_)
    # n,d = x.shape
    m, d = y.shape
    x_lo = lo(x)  # x_lo = numpy.array([xi.lo for xi in x])
    x_hi = hi(x)  # x_hi = numpy.array([xi.hi for xi in x])
    where_intersect = numpy.zeros((m,), dtype=bool)  # inter = []
    for i, yi in enumerate(tolist(y)):  #  a focal elem
        where_intersect[i] = any(
            numpy.all(~((x_hi < lo(yi)) | (hi(yi) < x_lo)), axis=1)
        )
    return where_intersect


#####################################################################################
# parser.py
#####################################################################################
# Universal parser


def intervalise(x_: Any, interval_index=-1) -> Union[Interval, Any]:
    """
    This function casts an array-like structure into an Interval structure.
    All array-like structures will be first coerced into an ndarray of floats.
    If the coercion is unsuccessful the following error is thrown: `ValueError: setting an array element with a sequence.`

    For example this is the expected behaviour:
    (*) an ndarray of shape (4,2) will be cast as an Interval of shape (4,).

    (*) an ndarray of shape (7,3,2) will be cast as an Interval of shape (7,3).

    (*) an ndarray of shape (3,2,7) will be cast as a degenerate Interval of shape (3,2,7).

    (*) an ndarray of shape (2,3,7) will be cast as an Interval of shape (3,7).

    (*) an ndarray of shape (2,3,7,2) will be cast as an Interval of shape (2,3,7) if interval_index is set to -1.

    If an ndarray has shape with multiple dimensions having size 2, then the last dimension is intervalised.
    So, an ndarray of shape (7,2,2) will be cast as an Interval of shape (7,2) with the last dimension intervalised.
    When the ndarray has shape (2,2) again is the last dimension that gets intervalised.

    In case of ambiguity, e.g. (2,5,2), now the first dimension can be forced to be intervalised, selecting index=0, default is -1.

    It returns an interval only if the input is an array-like structure, otherwise it returns the following numpy error:
    `ValueError: setting an array element with a sequence.`

    TODO: Parse a list of mixed numbers: interval and ndarrays.

    """

    def treat_list(xx):
        xi_lo, xi_hi = [], []
        for xi in xx:  # if each element in the list is an interval of homogeneus shape
            if xi.__class__.__name__ == "Interval":
                xi_lo.append(xi.lo)
                xi_hi.append(xi.hi)
            else:
                xi_ = intervalise(xi)  # recursion
                xi_lo.append(xi_.lo)
                xi_hi.append(xi_.hi)
        try:
            return Interval(lo=xi_lo, hi=xi_hi)
        except:
            print("!! Parsing an interval from list failed.")
            return x_

    if x_.__class__.__name__ == "Interval":
        return x_
    try:
        x = asarray(x_, dtype=float)
    # ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (...) + inhomogeneous part.
    except ValueError:
        if x_.__class__.__name__ == "list":
            # attempt to turn a n-list of intervals into a (n,...)-interval
            return treat_list(x_)
    s = x.shape
    two = [si == 2 for si in s]
    # if all(two): return Interval(lo=numpy_transpose(x)[0],hi=numpy_transpose(x)[1])
    if all(two):
        return Interval(lo=x[..., 0], hi=x[..., 1])
    elif any(two):
        # if two[-1]: return Interval(lo=numpy_transpose(x)[0],hi=numpy_transpose(x)[1]) # the last dimension has size 2
        if two[-1]:
            # last dimension has size 2
            return Interval(lo=x[..., 0], hi=x[..., 1])
        elif two[0]:
            return Interval(lo=x[0], hi=x[1])  # first dimension has size 2
        elif (two[-1]) & (two[0]):  # this is the ambiguous case (2,3,5,2)
            if interval_index == 0:
                # first dimension gets intervalised
                return Interval(lo=x[0], hi=x[1])
            # elif index == -1: return Interval(lo=numpy_transpose(x)[0],hi=numpy_transpose(x)[1])
            elif interval_index == -1:
                return Interval(lo=x[..., 0], hi=x[..., 1])
            # if (sum(two)==1) & (two[0]): return Interval(lo=x[0],hi=x[1])# there is only one dimension of size 2 and is the first one
        print(
            "Array-like structure must have the last (or first) dimension of size 2, for it to be coerced to Interval."
        )
        return Interval(lo=x)
    else:
        return Interval(lo=x)


def sizeit(x: Interval) -> Interval:
    """
    Takes an unsized scalar interval and turns it in to a sized one.
    """
    if is_Interval(x):
        if x.scalar & x.unsized:
            return Interval(lo=[x.lo], hi=[x.hi])
    return x


def unsizeit(x: Interval) -> Interval:
    """
    Takes a sized scalar interval and turns it in to a unsized one.

    """
    if is_Interval(x):
        if x.scalar & x.unsized == False:
            return Interval(lo=x.lo[0], hi=x.hi[0])
    return x


def tolist(x: Interval):
    if is_not_Interval(x):
        return x
    dim = len(x.shape)
    if dim == 1:
        return [Interval(xi.lo, xi.hi) for xi in x]
    if dim == 2:
        m, d = x.shape
        return [Interval(lo=x.lo[i, :], hi=x.hi[i, :]) for i in range(m)]
    if dim > 2:
        return x  # not implemented yet


#####################################################################################
# subint.py
#####################################################################################


def subintervalise(x_: Interval, n: Union[int, tuple] = 0) -> Interval:
    """
    return:
        - matrix Interval
    """
    x = intervalise(x_)
    d = len(x.shape)  # dimension of the array
    if n == 0 | n == 1:
        return x  # should return a subtiling (sized interval)
    if x.scalar:  # or x.scalar == True
        xx = linspace(x.lo, x.hi, num=n + 1)
        return intervalise(vstack([xx[:-1], xx[1:]]))
    elif d == 1:  # x.shape = (m,)
        m = x.shape[0]  # size of 1d array
        if type(n) == int:
            n = m * [n]  # differential split number
        X_sub = []
        for i, xi in enumerate(x):
            xxi = subintervalise(xi, n=n[i])  # recursion
            X_sub.append(sizeit(xxi).val)
        return intervalise(asarray(list(product(*X_sub)), dtype=float))
    #     elif len(x.shape)>1: pass # TODO: implement space-product subintervalization with arrays of dimension greater than 2.
    else:
        print(
            "!! Subtiling not yet supported for interval arrays of dimension 2 or larger. Input will be returned."
        )
    return x


def split_interval(x: Interval, y: float = None):
    if y is None:
        return Interval(lo(x), mid(x)), Interval(mid(x), hi(x))

    x1, x2 = Interval(lo(x), hi(x)), Interval(
        lo(x), hi(x)
    )  # TODO: implement copy method
    y_in_x = contain(x, y)  # x contain y
    if x.unsized:
        if ~y_in_x:
            return x1, x2
        return Interval(lo=lo(x), hi=y), Interval(lo=y, hi=hi(x))
    else:
        pass
        # x1[y_in_x] = Interval(lo(x)[y_in_x],hi=y)
    # x2[y_in_x] = Interval(y,hi=hi(x)[y_in_x])
    return x1, x2


def reconstitute(x_: Interval):
    x = intervalise(x_)
    d = len(x.shape)  #  dimension of the subtiling ==1 if scalar, ==2 if 1d array
    if d == 1:
        return Interval(lo=numpy.min(x.lo), hi=numpy.max(x.hi))
    elif d == 2:
        return Interval(lo=numpy.min(x.lo, axis=1), hi=numpy.max(x.hi, axis=1))
    else:
        print(
            "!! Subtiling not yet supported for interval arrays of dimension 2 or larger."
        )
    return x


def space_product(x_: Union[ndarray, Interval], y_: Union[ndarray, Interval]):
    return asarray(tuple(product(x_, y_)))


def bisect(x_: Interval, i: int = None):
    """
    :x_: Interval of shape (n,)

    Bisect the largest box if i is None.
    """
    x = intervalise(x_)
    if x.scalar:
        mid_x = mid(x)
        return Interval(lo(x), mid_x), Interval(mid_x, hi(x))
    if i is not None:
        split_index = i
    else:
        w = width(x)
        split_index = argmax(w)
    d = x.shape[0]
    n = [0] * d
    # ex: (0,0,2,0,0,0) if interval of dim 6 has third dimension bisected
    n[split_index] = 2
    x_bisect = subintervalise(x, n=tuple(n))
    return x_bisect[0], x_bisect[1]


#####################################################################################
# types.py
#####################################################################################
# Interval to bool methods, Unary.


def is_Interval(x: Any) -> bool:
    return x.__class__.__name__ == "Interval"


def is_not_Interval(x: Any) -> bool:
    return x.__class__.__name__ != "Interval"


#####################################################################################
################################# neural_networks.py ################################
#####################################################################################


def dot(x: Interval, y: Interval):
    return sum(x * y)


def rowcol_old(W, x):
    """
    (m,n) x (n,1) -> (m,1)
    (m,n) x (n,p) -> (m,p)
    (1,n) x (n,1) -> (1,1)
    """
    s = W.shape
    x_shape = x.shape
    if x_shape[0] == 1:  # x is row and must be either squeezed or transposed
        x_ = x[0, :]
    if x_shape[1] == 1:  # this is the correct shape
        x_ = x[:, 0]
    if len(x_shape) == 1:  # x can be row or column (n,)
        x_ = x
    y = []
    for i in range(s[0]):
        y.append(dot(W[i], x_))
    return intervalise(y)


def rowcol_W_x(W, x):
    """
    Row by column multiplication between a matrix W and a column vector x.

    (m,n) x (n,1) -> (m,1)
    (1,n) x (n,1) -> (1,1)
    The following cases are also accepted even though mathematically impossible
    (m,n) x (n,) -> (m,1)
    (1,n) x (n,1) -> (1,1)
    (1,n) x (1,n) -> (1,1)
    """
    m, n = W.shape
    x_shape = x.shape
    if not ((n == x_shape[0]) | (n == x_shape[1])):
        raise ValueError(
            f"Incompatible shapes [{m}, {n}] x({x_shape[0]}, 1) -> (?, ?). Inner sizes must be same, {x_shape[1]} is different from {n}."
        )
    if x_shape[0] == 1:  # x is row and must be either squeezed or transposed
        x_ = x[0, :]
    if x_shape[1] == 1:  # this is the correct shape
        x_ = x[:, 0]
    if len(x_shape) == 1:  # x can be row or column (n,)
        x_ = x
    y = numpy.empty((m, 2))
    for i in range(m):
        inner_product = dot(W[i], x_)
        y[i, 0] = inner_product.lo
        y[i, 1] = inner_product.hi
    ylo = numpy.expand_dims(y[..., 0], axis=1)
    yhi = numpy.expand_dims(y[..., 1], axis=1)
    return Interval(ylo, yhi)


def rowcol_xT_WT(x, W):
    """
    Row by column multiplication between the row vector xT and the matrix transpose WT.
    (1,n) x (n,m) -> (1,m)
    (1,n) x (n,1) -> (1,1)
    The following cases are also accepted even though mathematically impossible
    (,n) x (n,m) -> (1,m)
    (n,1) x (n,m) -> (1,1)
    """
    n, m = W.shape
    x_shape = x.shape
    if not ((n == x_shape[0]) | (n == x_shape[1])):
        raise ValueError(
            f"Incompatible shapes(1, {x_shape[1]}) x({n}, {m}) -> (?, ?). Inner sizes must be same, {x_shape[1]} is different from {n}."
        )
    if x_shape[0] == 1:  # this is the correct shape
        x_ = x[0, :]
    if x_shape[1] == 1:  # x is row and must be either squeezed or transposed
        x_ = x[:, 0]
    if len(x_shape) == 1:  # x can be row or column (n,)
        x_ = x
    y = numpy.empty((m, 2))
    for i in range(m):
        inner_product = dot(x_, W[..., i])
        y[i, 0] = inner_product.lo
        y[i, 1] = inner_product.hi
    ylo = numpy.expand_dims(y[..., 0], axis=0)
    yhi = numpy.expand_dims(y[..., 1], axis=0)
    return Interval(ylo, yhi)


def matmul(A, B):
    """
    (m,n) x (n,p) -> (m,p)
    (1,n) x (n,1) -> (1,1)
    """
    m, na = A.shape
    nb, p = B.shape
    if na != nb:
        raise ValueError(
            f"Incompatible shapes({m}, {na}) x({nb}, {p}) -> (?, ?). Inner sizes must be same, {na} is different from {nb}."
        )
    C = numpy.empty((m, p, 2))
    for i in range(m):
        for j in range(p):
            inner_product = dot(A[i], B[..., j])
            C[i, j, 0] = inner_product.lo
            C[i, j, 1] = inner_product.hi
    return intervalise(C)


def transpose(x: Interval):  # not efficient it creates a new object in memory
    """
    Input an interval of shape (m,n) returns an interval of shape (n,m).
    """
    return Interval(x.val[..., 0], x.val[..., 1])


def squeeze(x: Interval):  # not efficient it creates a new object in memory
    return Interval(numpy.squeeze(x.lo), numpy.squeeze(x.hi))


################################ activation_functions.py #############################


def relu_nointerval(x: ndarray):
    positive = x > 0
    output = numpy.zeros(x.shape)
    output[positive] = x[positive]
    return output


def relu_deriv(x: ndarray):
    positive = x > 0
    output = numpy.zeros(x.shape)
    output[positive] = 1
    return output


def relu(x: Interval):
    if is_not_Interval(x):
        return relu_nointerval(x)
    case_1 = x.hi < 0
    x_lo = x.val[..., 0].T
    x_hi = x.val[..., 1].T
    x_lo[case_1] = 0
    x_hi[case_1] = 0
    case_3 = (x_lo < 0) & (numpy.logical_not(case_1))
    x_lo[case_3] = 0
    relu_x = Interval(lo=x_lo, hi=x_hi)
    return relu_x


def relu_deriv_interval(x: Interval):
    if is_not_Interval(x):
        return relu_deriv(x)
    x_lo = x.val[..., 0].T
    x_hi = x.val[..., 1].T
    case_1 = x_hi < 0
    d_lo = numpy.zeros(x_lo.shape)
    d_hi = numpy.ones(x_hi.shape)
    d_hi[case_1] = 0
    # case_3 = (x_lo<0) & (np.logical_not(case_1))
    case_2 = x_lo > 0
    d_lo[case_2] = 1
    d_relu_x = Interval(lo=d_lo, hi=d_hi)
    return d_relu_x


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


# def tanh_(x): return np.tanh(x)


def tanh(x: Interval):
    return (exp(2 * x) - 1) / (exp(2 * x) + 1)


def tanh(x: Interval):
    r = -1
    s = 1
    u = 1
    t = 1
    return s / u - (s * t / u**2 - r / u) / (t / u + exp(2 * x))


# def cot(x): return 1/np.tan(x)
# def tanh(x): return -(1/(cot(np.arctan(x)/2)**2))
def cosh(x: Interval):
    return (1 + exp(-2 * x)) / (2 * exp(-x))


def tanh_deriv(x: Interval):
    return (1 / cosh(x)) ** 2
