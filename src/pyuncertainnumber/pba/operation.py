from __future__ import annotations
from typing import TYPE_CHECKING
import itertools
import functools
import numpy as np
from numbers import Number
from .params import Params
import operator

if TYPE_CHECKING:
    from .pbox_abc import Pbox


# * ---------------  Frechet ops --------------- *#


def frechet_op(x: Pbox, y: Pbox, op=operator.add):
    """Frechet operation on two pboxes

    note:
        this corresponds to the Frank, Nelson and Sklar Frechet bounds implementation
    """

    assert x.steps == y.steps, "Pboxes must have the same number of steps"

    n = x.steps

    nleft = np.empty(n)
    nright = np.empty(n)

    for i in range(0, n):
        j = np.arange(i, n)
        k = np.arange(n - 1, i - 1, -1)
        nright[i] = np.min(op(x.right[j], y.right[k]))
        jj = np.arange(0, i + 1)
        kk = np.arange(i, -1, -1)
        nleft[i] = np.max(op(x.left[jj], y.left[kk]))

    nleft.sort()
    nright.sort()

    return nleft, nright


def naive_frechet_op(x: Pbox, y: Pbox, op):
    """The real naive Frechet bounds implementation

    note:
        counterpart of `naivefrechetpbox` in `pba.r`

    args:
        x, y (Pbox): pboxes to be operated on
        op (function): arithemtic operator to be applied, e.g. operator.add

    return:
        Zu, Zd (tuple): left and right bounds of the pbox
        - Zu: lower bound of the pbox
        - Zd: upper bound of the pbox

    example:
        >>> from pyuncertainnumber import pba
        >>> a = pba.normal([4,5], 1)
        >>> b = pba.uniform([3,4], [7,8])
        >>> left_bound, right_bound = pba.naive_frechet_op(a, b, operator.add)
    """

    assert x.steps == y.steps, "Pboxes must have the same number of steps"

    n = x.steps
    # right
    c_r = vectorized_cartesian_op(x.right, y.right, op)
    c_r.sort()
    Zd = c_r[(n * n - n) : (n * n)]

    # left
    c = vectorized_cartesian_op(x.left, y.left, op)
    c.sort()
    Zu = c[:n]
    return Zu, Zd


def new_naive_frechet_op(
    x: Pbox,
    y: Pbox,
    op,
    n_sam=Params.steps,
):
    """this is a slow version"""
    from functools import partial
    from pyuncertainnumber import b2b, make_vec_interval

    # p_vars = [convert_pbox(v) for v in vars]

    n = n_sam  # number of samples to be taken

    assert x.steps == y.steps, "Pboxes must have the same number of steps"

    # if n < x.steps:
    #     # this change when there's specified dependency structure
    #     alpha = np.squeeze(qmc.LatinHypercube(d=1).random(n=n_sam))
    #     itvs = [v.alpha_cut(alpha) for v in [x, y]]
    # else:
    itvs = [v.discretise(n_sam) for v in [x, y]]

    # TODO add parallel logic herein
    b2b_f = partial(b2b, func=lambda x: op(x[0], x[1]), interval_strategy="direct")
    container = [b2b_f(_item) for _item in itertools.product(*itvs)]
    container = make_vec_interval(container)

    Zu = np.sort(container.lo)  # left
    Zd = np.sort(container.hi)  # right
    Zu = Zu[:n]  # take the first n elements
    Zd = Zd[(n * n - n) : (n * n)]  # take the last n elements
    return Zu, Zd


# def new_vectorised_naive_frechet_op(x, y, op):
#     """The new vectorised Frechet ops"""
#     assert x.steps == y.steps, "Pboxes must have the same number of steps"
#     n = x.steps
#     Xd = np.tile(x.right, n)
#     Xu = np.tile(x.left, n)
#     Yd = np.tile(y.right, n)
#     Yu = np.tile(y.left, n)

#     c1 = op(Xu, Yu)
#     c2 = op(Xu, Yd)
#     c3 = op(Xd, Yu)
#     c4 = op(Xd, Yd)
#     Zu = np.sort(np.minimum.reduce([c1, c2, c3, c4]))
#     Zd = np.sort(np.maximum.reduce([c1, c2, c3, c4]))

#     # I believe there should not be condensation per se, but rather the following boundings
#     Zu = Zu[:n]
#     Zd = Zd[(n * n - n) : (n * n)]
#     return Zu, Zd


def perfect_op(x: Pbox, y: Pbox, op=operator.add):
    """perfect operation on two pboxes

    note:
        defined for addition and multiplication. Different for subtraction and division.
    """
    nleft = op(x.left, y.left)
    nright = op(x.right, y.right)

    nleft.sort()
    nright.sort()
    return nleft, nright


def opposite_op(x: Pbox, y: Pbox, op=operator.add):
    """opposite operation on two pboxes

    note:
        defined for addition and multiplication. Different for subtraction and division.
    """
    nleft = op(x.left, np.flip(y.left))
    nright = op(x.right, np.flip(y.right))
    nleft.sort()
    nright.sort()
    return nleft, nright


def independent_op(x: Pbox, y: Pbox, op=operator.add):
    """independent operation on two pboxes

    note:
        defined for addition and multiplication. Different for subtraction and division.
    """
    c1 = vectorized_cartesian_op(x.left, y.left, op)
    c2 = vectorized_cartesian_op(x.left, y.right, op)
    c3 = vectorized_cartesian_op(x.right, y.left, op)
    c4 = vectorized_cartesian_op(x.right, y.right, op)

    nleft = np.sort(np.minimum.reduce([c1, c2, c3, c4]))
    nright = np.sort(np.maximum.reduce([c1, c2, c3, c4]))

    return nleft, nright


# backup
# def independent_op(x: Pbox, y: Pbox, op=operator.add):
#     """independent operation on two pboxes

#     note:
#         defined for addition and multiplication. Different for subtraction and division.
#     """
#     nleft = vectorized_cartesian_op(x.left, y.left, op)
#     nright = vectorized_cartesian_op(x.right, y.right, op)
#     return nleft, nright


def new_vectorised_naive_frechet_op(x: Pbox, y: Pbox, op):
    """independent operation on two pboxes

    note:
        defined for addition and multiplication. Different for subtraction and division.
    """
    n = x.steps

    c1 = vectorized_cartesian_op(x.left, y.left, op)
    c2 = vectorized_cartesian_op(x.left, y.right, op)
    c3 = vectorized_cartesian_op(x.right, y.left, op)
    c4 = vectorized_cartesian_op(x.right, y.right, op)

    Zu = np.sort(np.minimum.reduce([c1, c2, c3, c4]))
    Zd = np.sort(np.maximum.reduce([c1, c2, c3, c4]))

    Zu = Zu[:n]
    Zd = Zd[(n * n - n) : (n * n)]
    return Zu, Zd
    # return c1, c2, c3, c4


def vectorized_cartesian_op(a, b, op):
    """vectorised cartesian operation for the bounds of pboxes

    note:
        used on self.left, other.left

    example:
        >>> nleft = vectorized_cartesian_op(self.left, other.left, operator.add)
        >>> nright = vectorized_cartesian_op(self.right, other.right, operator.add)
    """
    return op(a[:, np.newaxis], b).ravel()


def isum(l_p):
    """Sum of pboxes indepedently

    args:
        l_p (list): list of Pbox objects

    note:
        Same signature with Python ``sum`` which takes a list of inputs

    tip:
        Python ``sum`` accomplishes sum of Frechet case.
    """

    def binary_independent_sum(p1, p2):
        return p1.add(p2, dependency="i")

    return functools.reduce(binary_independent_sum, l_p)


# there is an new `convert` func
def convert(un):
    """Convert any input un into a Pbox object

    note:
        - theorically 'un' can be {Interval, DempsterShafer, Distribution, float, int}
    """

    from .pbox_abc import Pbox
    from .dss import DempsterShafer
    from .distributions import Distribution
    from .intervals.number import Interval

    if isinstance(un, Pbox):
        return un
    elif isinstance(un, Interval):
        return un.to_pbox()
    elif isinstance(un, Distribution):
        return un.to_pbox()
    elif isinstance(un, DempsterShafer):
        return un.to_pbox()
    elif isinstance(un, Number):
        return Interval(un, un).to_pbox()
    else:
        raise TypeError(f"Unable to convert {type(un)} object to Pbox")


def p_backcalc(a, c, ops):
    """backcal for p-boxes
    #! incorrect implementation
    args:
        a, c (Pbox):probability box objects
        ops (object) : {'additive_bcc', 'multiplicative_bcc'} whether additive or multiplicative
    """
    from pyuncertainnumber.pba.intervals.intervalOperators import make_vec_interval
    from pyuncertainnumber.pba.aggregation import stacking
    from .pbox_abc import Pbox
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


# * --------------- independent ops --------------- *#


def i_mul(a, b):
    """bivariate independent multiplication of pboxes"""

    if isinstance(a, Number) or isinstance(b, Number):
        return a * b
    else:  # 2 pboxes
        a, b = convert(a), convert(b)
        return a.mul(b, dependency="i")


# * --------------- vectorisation --------------- *#

###### base implementation of vector and matrix operations succeeds


class Vector:
    def __init__(self, components):
        self.components = components

    def __iter__(self):
        return iter(self.components)

    def __len__(self):
        return len(self.components)

    def __repr__(self):
        return f"Vector({self.components})"

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Vector(self.components[index])
        else:
            return self.components[index]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            if isinstance(value, Vector):
                self.components[index] = value.components
            elif isinstance(value, list):
                self.components[index] = value
            else:
                raise TypeError(
                    "Assigned value must be a Vector or list for slice assignment"
                )
        else:
            self.components[index] = value

    def __add__(self, other):
        if isinstance(other, Vector | list):
            if len(self) != len(other):
                raise ValueError("Vectors must be the same length")
            return Vector([a + b for a, b in zip(self, other)])
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be the same length")
            return Vector([a - b for a, b in zip(self, other)])
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector([x * other for x in self.components])
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector([x / other for x in self.components])
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, Vector | list):
            if len(self) != len(other):
                raise ValueError("Vectors must be the same length")
            return sum(x * y for x, y in zip(self, other))

        elif isinstance(other, Matrix):
            # Vector @ Matrix: treat self as row vector, multiply by matrix
            n = len(self)
            m_rows, m_cols = other.shape()
            if n != m_rows:
                raise ValueError(
                    "Vector length must match matrix rows (row vector @ matrix)"
                )
            result = []
            for col in zip(*other.rows):
                result.append(sum(v * c for v, c in zip(self.components, col)))
            return Vector(result)

        return NotImplemented

    def tanh(self):
        """Pbox tanh operation elemenwise"""
        return Vector([x.tanh() for x in self.components])


class Matrix:
    def __init__(self, rows):
        # Handle case where input is a single Vector
        if isinstance(rows, Vector):
            # Turn Vector into a single-row Matrix
            self.rows = [list(rows.components)]
        elif isinstance(rows, list | np.ndarray):
            # Check if each row is a Vector, if so convert to list
            processed_rows = []
            for row in rows:
                if isinstance(row, Vector):
                    processed_rows.append(list(row.components))
                elif isinstance(row, list | np.ndarray):
                    processed_rows.append(row)
                else:
                    raise TypeError(f"Unsupported row type: {type(row)}")

            # Verify all rows have the same length
            if not all(len(row) == len(processed_rows[0]) for row in processed_rows):
                raise ValueError("All rows must have the same length")

            self.rows = processed_rows
        else:
            raise TypeError(f"Unsupported input type: {type(rows)}")

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)

    def shape(self):
        return len(self.rows), len(self.rows[0])

    def __repr__(self):
        return f"Matrix({self.rows})"

    def __getitem__(self, index):
        if isinstance(index, tuple):
            # Expecting (row, col)
            row, col = index
            return self.rows[row][col]
        elif isinstance(index, slice):
            # Slice rows, return a new Matrix
            return Matrix(self.rows[index])
        else:
            # Single row, return it as Vector or list; here I use Vector for consistency
            return Vector(self.rows[index])

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            # index is (row, col)
            row, col = index
            self.rows[row][col] = value

        elif isinstance(index, slice):
            # index is a slice of rows
            if isinstance(value, Matrix):
                self.rows[index] = value.rows
            elif isinstance(value, list):
                self.rows[index] = value
            else:
                raise TypeError(
                    "Assigned value must be a Matrix or list of rows for slice assignment"
                )

        else:
            # Single row assignment
            if isinstance(value, Vector):
                self.rows[index] = value.components
            elif isinstance(value, list):
                self.rows[index] = value
            else:
                raise TypeError(
                    "Assigned value must be a Vector or list for row assignment"
                )

    def __add__(self, other):
        if isinstance(other, Matrix):
            if self.shape() != other.shape():
                raise ValueError("Matrices must have the same shape")
            return Matrix(
                [
                    [a + b for a, b in zip(row1, row2)]
                    for row1, row2 in zip(self.rows, other.rows)
                ]
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Matrix):
            if self.shape() != other.shape():
                raise ValueError("Matrices must have the same shape")
            return Matrix(
                [
                    [a - b for a, b in zip(row1, row2)]
                    for row1, row2 in zip(self.rows, other.rows)
                ]
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[x * other for x in row] for row in self.rows])
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[x / other for x in row] for row in self.rows])
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, Vector):
            # Matrix @ Vector
            m, n = self.shape()
            if n != len(other):
                raise ValueError("Matrix columns must match vector length")
            return Vector(
                [sum(r[i] * other.components[i] for i in range(n)) for r in self.rows]
            )

        elif isinstance(other, Matrix):
            # Matrix @ Matrix
            m1, n1 = self.shape()
            m2, n2 = other.shape()
            if n1 != m2:
                raise ValueError("Incompatible shapes for matrix multiplication")
            result_rows = []
            for row in self.rows:
                result_row = []
                for col in zip(*other.rows):
                    result_row.append(sum(r * c for r, c in zip(row, col)))
                result_rows.append(result_row)
            return Matrix(result_rows)

        return NotImplemented

    def tanh(self):
        return Matrix([[x.tanh() for x in row] for row in self.rows])


# * --------------- independet Vector and Matrix for Pbox operations --------------- *#
class iVector(Vector):
    """Independent Vector of Pboxes"""

    def __init__(self, components):
        super().__init__(components)

    def __add__(self, other):
        if isinstance(other, iVector | list):
            if len(self) != len(other):
                raise ValueError("Vectors must be the same length")
            return iVector([isum(a, b) for a, b in zip(self, other)])
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, iVector | list):
            if len(self) != len(other):
                raise ValueError("Vectors must be the same length")
            return isum([i_mul(x, y) for x, y in zip(self, other)])

        elif isinstance(other, Matrix):
            # Vector @ Matrix: treat self as row vector, multiply by matrix
            n = len(self)
            m_rows, m_cols = other.shape()
            if n != m_rows:
                raise ValueError(
                    "Vector length must match matrix rows (row vector @ matrix)"
                )
            result = []
            for col in zip(*other.rows):
                result.append(isum(i_mul(v, c) for v, c in zip(self.components, col)))
            return Vector(result)

        return NotImplemented


class iMatrix(Matrix):
    """Independent Matrix of Pboxes"""

    def __init__(self, rows):
        super().__init__(rows)

    def __add__(self, other):
        if isinstance(other, iMatrix):
            if self.shape() != other.shape():
                raise ValueError("Matrices must have the same shape")
            return iMatrix(
                [
                    [isum(a, b) for a, b in zip(row1, row2)]
                    for row1, row2 in zip(self.rows, other.rows)
                ]
            )
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, iVector):
            # Matrix @ Vector
            m, n = self.shape()
            if n != len(other):
                raise ValueError("Matrix columns must match vector length")
            return iVector(
                [
                    isum(i_mul(r[i], other.components[i]) for i in range(n))
                    for r in self.rows
                ]
            )

        elif isinstance(other, iMatrix):
            # Matrix @ Matrix
            m1, n1 = self.shape()
            m2, n2 = other.shape()
            if n1 != m2:
                raise ValueError("Incompatible shapes for matrix multiplication")
            result_rows = []
            for row in self.rows:
                result_row = []
                for col in zip(*other.rows):
                    result_row.append(isum(i_mul(r, c) for r, c in zip(row, col)))
                result_rows.append(result_row)
            return iMatrix(result_rows)

        return NotImplemented


### use this later as `x.mul(y, dependency="i")` affects generality with other data types
class Vector_pbox(Vector):
    """Vector of Pboxes"""

    def __init__(self, components):
        super().__init__(components)

    def __matmul__(self, other):
        if isinstance(other, Vector | list):
            if len(self) != len(other):
                raise ValueError("Vectors must be the same length")
            return sum(x.mul(y, dependency="i") for x, y in zip(self, other))

        elif isinstance(other, Matrix):
            # Vector @ Matrix: treat self as row vector, multiply by matrix
            n = len(self)
            m_rows, m_cols = other.shape()
            if n != m_rows:
                raise ValueError(
                    "Vector length must match matrix rows (row vector @ matrix)"
                )
            result = []
            for col in zip(*other.rows):
                result.append(
                    sum(v.mul(c, dependency="i") for v, c in zip(self.components, col))
                )
            return Vector(result)

        return NotImplemented


#### backup ####
# class Matrix:
#     def __init__(self, rows):
#         if not all(len(row) == len(rows[0]) for row in rows):
#             raise ValueError("All rows must have the same length")
#         self.rows = rows

#     def __getitem__(self, index):
#         return self.rows[index]

#     def __len__(self):
#         return len(self.rows)

#     def shape(self):
#         return len(self.rows), len(self.rows[0])

#     def __repr__(self):
#         return f"Matrix({self.rows})"

#     def __add__(self, other):
#         if isinstance(other, Matrix):
#             if self.shape() != other.shape():
#                 raise ValueError("Matrices must have the same shape")
#             return Matrix(
#                 [
#                     [a + b for a, b in zip(row1, row2)]
#                     for row1, row2 in zip(self.rows, other.rows)
#                 ]
#             )
#         return NotImplemented

#     def __sub__(self, other):
#         if isinstance(other, Matrix):
#             if self.shape() != other.shape():
#                 raise ValueError("Matrices must have the same shape")
#             return Matrix(
#                 [
#                     [a - b for a, b in zip(row1, row2)]
#                     for row1, row2 in zip(self.rows, other.rows)
#                 ]
#             )
#         return NotImplemented

#     def __mul__(self, other):
#         if isinstance(other, (int, float)):
#             return Matrix([[x * other for x in row] for row in self.rows])
#         return NotImplemented

#     def __rmul__(self, other):
#         return self * other

#     def __truediv__(self, other):
#         if isinstance(other, (int, float)):
#             return Matrix([[x / other for x in row] for row in self.rows])
#         return NotImplemented

#     def __matmul__(self, other):
#         if isinstance(other, Vector):
#             # Matrix @ Vector
#             m, n = self.shape()
#             if n != len(other):
#                 raise ValueError("Matrix columns must match vector length")
#             return Vector(
#                 [sum(r[i] * other.components[i] for i in range(n)) for r in self.rows]
#             )

#         elif isinstance(other, Matrix):
#             # Matrix @ Matrix
#             m1, n1 = self.shape()
#             m2, n2 = other.shape()
#             if n1 != m2:
#                 raise ValueError("Incompatible shapes for matrix multiplication")
#             result_rows = []
#             for row in self.rows:
#                 result_row = []
#                 for col in zip(*other.rows):
#                     result_row.append(sum(r * c for r, c in zip(row, col)))
#                 result_rows.append(result_row)
#             return Matrix(result_rows)

#         return NotImplemented
