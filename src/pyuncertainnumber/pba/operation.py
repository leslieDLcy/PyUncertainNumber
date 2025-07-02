from __future__ import annotations
from typing import TYPE_CHECKING
import itertools
import functools
import numpy as np
from numbers import Number
from .params import Params

if TYPE_CHECKING:
    from .pbox_abc import Pbox


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
    """transform the input un into a Pbox object

    note:
        - theorically 'un' can be {Interval, DempsterShafer, Distribution, float, int}
    """

    from .pbox_abc import Pbox
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


class Matrix:
    def __init__(self, rows):
        if not all(len(row) == len(rows[0]) for row in rows):
            raise ValueError("All rows must have the same length")
        self.rows = rows

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)

    def shape(self):
        return len(self.rows), len(self.rows[0])

    def __repr__(self):
        return f"Matrix({self.rows})"

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
