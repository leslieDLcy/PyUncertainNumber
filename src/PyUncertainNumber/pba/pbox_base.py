from .operation import convert
from decimal import DivisionByZero
from typing import *
from warnings import *

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from .interval import Interval as nInterval
from .utils import find_nearest, check_increasing, NotIncreasingError, _interval_list_to_array
from .params import Params
import importlib

__all__ = [
    "Pbox",
    "mixture",
    "truncate",
    "imposition",
    "NotIncreasingError",
]

""" Nick: It is usually better to define p-boxes using distributions or non-parametric methods (see ). 
This constructor is provided for completeness and for the construction of p-boxes with precise inputs.

Leslie: the above does not make sense. The Pbox class is the base constructor for p-boxes, which has been
called by whatever other constructors currently exist. The above comment is not accurate.
"""


class Pbox:
    """ the base constructor """

    r"""
    A probability distribution is a mathematical function that gives the probabilities of occurrence for diﬀerent possible values of a variable. Probability boxes (p-boxes) represent interval bounds on probability distributions. The simplest kind of p-box can be expressed mathematically as

    .. math::

        \mathcal{F}(x) = [\underline{F}(x),\overline{F}(x)], \ \underline{F}(x)\geq \overline{F}(x)\ \forall x \in \mathbb{R}


    where :math:`\underline{F}(x)` is the function that defines the left bound of the p-box and :math:`\overline{F}(x)` defines the right bound of the p-box. In PBA the left and right bounds are each stored as a NumPy array containing the percent point function (the inverse of the cumulative distribution function) for `steps` evenly spaced values between 0 and 1. P-boxes can be defined using all the probability distributions that are available through SciPy's statistics library,

    Naturally, precise probability distributions can be defined in PBA by defining a p-box with precise inputs. This means that within probability bounds analysis probability distributions are considered a special case of a p-box with zero width. Resultantly, all methodology that applies to p-boxes can also be applied to probability distributions.

    Distribution-free p-boxes can also be generated when the underlying distribution is unknown but parameters such as the mean, variance or minimum/maximum bounds are known. Such p-boxes make no assumption about the shape of the distribution and instead return bounds expressing all possible distributions that are valid given the known information. Such p-boxes can be constructed making use of Chebyshev, Markov and Cantelli inequalities from probability theory.
    """

    def __init__(
        self,
        left=None,
        right=None,
        steps=None,
        shape=None,
        mean_left=None,
        mean_right=None,
        var_left=None,
        var_right=None,
        interpolation="linear",
    ):
        """
        args:
            - left (list, NumPy array, nInterval or numeric type): Left bound of the p-box.

        note:
            - If left is None, the left bound is set to -inf.
        """
        self.shape = shape

        def init_check(left, right, steps):
            """ initialisation logic """

            if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                if len(left) != len(right):
                    raise Exception(
                        "Left and right arrays must be the same length")
                else:
                    steps = len(left)

            if steps is None:
                if hasattr(left, "__len__"):
                    steps = len(left)
                elif hasattr(right, "__len__"):
                    steps = len(right)
                else:
                    steps = Params.steps

            if (left is not None) and (right is None):
                right = left

            if left is None and right is None:
                left = np.array([-np.inf] * steps)
                right = np.array([np.inf] * steps)

            if isinstance(left, nInterval):
                left = np.array([left.left] * steps)
            elif isinstance(left, list):
                left = _interval_list_to_array(left)
            elif not isinstance(left, np.ndarray):
                left = np.array([left] * steps)

            if isinstance(right, nInterval):
                right = np.array([right.right] * steps)
            elif isinstance(right, list):
                right = _interval_list_to_array(right, left=False)
            elif not isinstance(right, np.ndarray):
                right = np.array([right] * steps)

            # if len(left) == len(right) and len(left) != steps:
            #     print("WARNING: The left and right arrays have the same length which is inconsistent with steps.")

            if len(left) != steps:
                left = _interpolate(
                    left, interpolation=interpolation, left=False, steps=steps
                )

            if len(right) != steps:
                right = _interpolate(
                    right, interpolation=interpolation, left=True, steps=steps
                )

            if not check_increasing(left) or not check_increasing(right):
                raise NotIncreasingError(
                    "Left and right arrays must be increasing")

            l, r = zip(*[(min(i), max(i)) for i in zip(left, right)])
            self.left = np.array(l)
            self.right = np.array(r)
            self.steps = steps
            self.mean_left = -np.inf
            self.mean_right = np.inf
            self.var_left = 0
            self.var_right = np.inf
        init_check(left, right, steps)

        if mean_left is not None:
            self.mean_left = np.max([mean_left, self.mean_left])
        if mean_right is not None:
            self.mean_right = np.min([mean_right, self.mean_right])
        if var_left is not None:
            self.var_left = np.max([var_left, self.var_left])
        if var_right is not None:
            self.var_right = np.min([var_right, self.var_right])

        self._computemoments()
        self._checkmoments()
        self.get_range()

    def __repr__(self):

        if self.mean_left == self.mean_right:
            mean_text = f"{round(self.mean_left, 4)}"
        else:
            mean_text = f"[{round(self.mean_left, 4)}, {round(self.mean_right, 4)}]"

        if self.var_left == self.var_right:
            var_text = f"{round(self.var_left, 4)}"
        else:
            var_text = f"[{round(self.var_left, 4)}, {round(self.var_right, 4)}]"

        range_text = [f"{val:.2f}" for val in self._range_list]

        if self.shape is None:
            shape_text = " "
        else:
            # space to start; see below lacking space
            shape_text = f" {self.shape}"

        return (
            f"Pbox: ~{shape_text}(range={range_text}, mean={mean_text}, var={var_text})"
        )

    __str__ = __repr__

    def __iter__(self):
        for val in np.array([self.left, self.right]).flatten():
            yield val

    # ---------------------Basic arithmetics---------------------#

    def __neg__(self):
        if self.shape in ["uniform", "normal", "cauchy", "triangular", "skew-normal"]:
            s = self.shape
        else:
            s = ""

        return Pbox(
            left=sorted(-np.flip(self.right)),
            right=sorted(-np.flip(self.left)),
            steps=len(self.left),
            shape=s,
            mean_left=-self.mean_right,
            mean_right=-self.mean_left,
            var_left=self.var_left,
            var_right=self.var_right,
        )

    def __lt__(self, other):
        return self.lt(other, method="f")

    def __rlt__(self, other):
        return self.ge(other, method="f")

    def __le__(self, other):
        return self.le(other, method="f")

    def __rle__(self, other):
        return self.gt(other, method="f")

    def __gt__(self, other):
        return self.gt(other, method="f")

    def __rgt__(self, other):
        return self.le(other, method="f")

    def __ge__(self, other):
        return self.ge(other, method="f")

    def __rge__(self, other):
        return self.lt(other, method="f")

    def __and__(self, other):
        return self.logicaland(other, method="f")

    def __rand__(self, other):
        return self.logicaland(other, method="f")

    def __or__(self, other):
        return self.logicalor(other, method="f")

    def __ror__(self, other):
        return self.logicalor(other, method="f")

    def __add__(self, other):
        return self.add(other, method="f")

    def __radd__(self, other):
        return self.add(other, method="f")

    def __sub__(self, other):
        return self.sub(other, method="f")

    def __rsub__(self, other):
        self = -self
        return self.add(other, method="f")

    def __mul__(self, other):
        return self.mul(other, method="f")

    def __rmul__(self, other):
        return self.mul(other, method="f")

    def __pow__(self, other):
        return self.pow(other, method="f")

    def __rpow__(self, other):
        if not hasattr(other, "__iter__"):
            other = np.array((other))

        b = Pbox(other)
        return b.pow(self, method="f")

    def __truediv__(self, other):

        return self.div(other, method="f")

    def __rtruediv__(self, other):

        try:
            return other * self.recip()
        except:
            return NotImplemented

    # * ---------------------properties---------------------*#
    @property
    def range(self):
        """leslie defined range property"""
        return self._range_list

    @property
    def lo(self):
        """
        Returns the left-most value in the interval
        """
        return self.left[0]

    def hi(self):
        """
        Returns the right-most value in the interval
        """
        return self.right[-1]

    # * --------------------- methods---------------------*#

    def get_range(self):
        """get the quantile range of either a pbox or a distribution"""
        self._range_list = [
            np.min([self.left, self.right]).tolist(),
            np.max([self.left, self.right]).tolist(),
        ]

    def _computemoments(
        self,
    ):  # should we compute mean if it is a Cauchy, var if it's a t distribution?
        self.mean_left = np.max([self.mean_left, np.mean(self.left)])
        self.mean_right = np.min([self.mean_right, np.mean(self.right)])

        if not (
            np.any(np.array(self.left) <= -np.inf)
            or np.any(np.inf <= np.array(self.right))
        ):
            V, JJ = 0, 0
            j = np.array(range(self.steps))

            for J in np.array(range(self.steps)) - 1:
                ud = [*self.left[j < J], *self.right[J <= j]]
                v = _sideVariance(ud)

                if V < v:
                    JJ = J
                    V = v

            self.var_right = V

    def _checkmoments(self):

        a = nInterval(self.mean_left, self.mean_right)  # mean(x)
        b = _dwMean(self)

        self.mean_left = np.max([left(a), left(b)])
        self.mean_right = np.min([right(a), right(b)])

        if self.mean_right < self.mean_left:
            # use the observed mean
            self.mean_left = left(b)
            self.mean_right = right(b)

        a = nInterval(self.var_left, self.var_right)  # var(x)
        b = _dwMean(self)

        self.var_left = np.max([left(a), left(b)])
        self.var_right = np.min([right(a), right(b)])

        if self.var_right < self.var_left:
            # use the observed variance
            self.var_left = left(b)
            self.var_right = right(b)

    # * ---------------------dual interpretation ---------------------#

    def cutv(self, x):
        """ get the bounds on the cumulative probability associated with any x-value """
        lo_ind = find_nearest(self.right, x)
        hi_ind = find_nearest(self.left, x)
        return nInterval(Params.p_values[lo_ind], Params.p_values[hi_ind])

    def cuth(self, p=0.5):
        """ get the bounds on the x-value at any particular probability level"""
        ind = find_nearest(Params.p_values, p)
        return nInterval(self.left[ind], self.right[ind])


# ---------------------unary operations---------------------#
    ##### the top-level functions for unary operations #####

    def _unary(self, *args, function=lambda x: x):

        ints = [function(nInterval(l, r), *args)
                for l, r in zip(self.left, self.right)]
        print("'referred to ints'", ints)
        return Pbox(
            left=np.array([i.left for i in ints]),
            right=np.array([i.right for i in ints]),
        )

    # Access Functions
    def add(self, other: Union["Pbox", nInterval, float, int], method="f") -> "Pbox":
        """
        Adds to Pbox to other using the defined dependency method


        """
        if method not in ["f", "p", "o", "i"]:
            raise ArithmeticError("Calculation method unkown")

        if other.__class__.__name__ == "nInterval":
            other = Pbox(other, steps=self.steps)

        if other.__class__.__name__ == "Pbox":

            if self.steps != other.steps:
                raise ArithmeticError(
                    "Both Pboxes must have the same number of steps")

            if method == "f":

                nleft = np.empty(self.steps)
                nright = np.empty(self.steps)

                for i in range(0, self.steps):
                    j = np.array(range(i, self.steps))
                    k = np.array(range(self.steps - 1, i - 1, -1))

                    nright[i] = np.min(self.right[j] + other.right[k])

                    jj = np.array(range(0, i + 1))
                    kk = np.array(range(i, -1, -1))

                    nleft[i] = np.max(self.left[jj] + other.left[kk])

            elif method == "p":

                nleft = self.left + other.left
                nright = self.right + other.right

            elif method == "o":

                nleft = self.left + np.flip(other.right)
                nright = self.right + np.flip(other.left)

            elif method == "i":

                nleft = []
                nright = []
                for i in self.left:
                    for j in other.left:
                        nleft.append(i + j)
                for ii in self.right:
                    for jj in other.right:
                        nright.append(ii + jj)

            nleft.sort()
            nright.sort()

            return Pbox(left=nleft, right=nright, steps=self.steps)

        else:
            try:
                # Try adding constant
                if self.shape in [
                    "uniform",
                    "normal",
                    "cauchy",
                    "triangular",
                    "skew-normal",
                ]:
                    s = self.shape
                else:
                    s = ""

                return Pbox(
                    left=self.left + other,
                    right=self.right + other,
                    shape=s,
                    mean_left=self.mean_left + other,
                    mean_right=self.mean_right + other,
                    var_left=self.var_left,
                    var_right=self.var_right,
                    steps=self.steps,
                )

            except:
                return NotImplemented

    def pow(self, other: Union["Pbox", nInterval, float, int], method="f") -> "Pbox":
        """
        Raises a p-box to the power of other using the defined dependency method

        :param other: Pbox, nInterval or numeric type
        :param method:

        :return: Pbox
        :rtype: Pbox

        """
        if method not in ["f", "p", "o", "i"]:
            raise ArithmeticError("Calculation method unkown")

        if other.__class__.__name__ == "nInterval":
            other = Pbox(other, steps=self.steps)

        if other.__class__.__name__ == "Pbox":

            if self.steps != other.steps:
                raise ArithmeticError(
                    "Both Pboxes must have the same number of steps")

            if method == "f":

                nleft = np.empty(self.steps)
                nright = np.empty(self.steps)

                for i in range(0, self.steps):
                    j = np.array(range(i, self.steps))
                    k = np.array(range(self.steps - 1, i - 1, -1))

                    nright[i] = np.min(self.right[j] ** other.right[k])

                    jj = np.array(range(0, i + 1))
                    kk = np.array(range(i, -1, -1))

                    nleft[i] = np.max(self.left[jj] ** other.left[kk])

            elif method == "p":

                nleft = self.left**other.left
                nright = self.right**other.right

            elif method == "o":

                nleft = self.left ** np.flip(other.right)
                nright = self.right ** np.flip(other.left)

            elif method == "i":

                nleft = []
                nright = []
                for i in self.left:
                    for j in other.left:
                        nleft.append(i + j)
                for ii in self.right:
                    for jj in other.right:
                        nright.append(ii + jj)

            nleft.sort()
            nright.sort()

            return Pbox(left=nleft, right=nright, steps=self.steps)

        else:
            try:
                # Try adding constant
                if self.shape in [
                    "uniform",
                    "normal",
                    "cauchy",
                    "triangular",
                    "skew-normal",
                ]:
                    s = self.shape
                else:
                    s = ""

                return Pbox(
                    left=self.left**other,
                    right=self.right**other,
                    shape=s,
                    mean_left=self.mean_left**other,
                    mean_right=self.mean_right**other,
                    var_left=self.var_left,
                    var_right=self.var_right,
                    steps=self.steps,
                )

            except:
                return NotImplemented

    def sub(self, other, method="f"):

        if method == "o":
            method = "p"
        elif method == "p":
            method = "o"

        return self.add(-other, method)

    def mul(self, other, method="f"):

        if method not in ["f", "p", "o", "i"]:
            raise ArithmeticError("Calculation method unkown")

        if other.__class__.__name__ == "nInterval":
            other = Pbox(other, steps=self.steps)

        if other.__class__.__name__ == "Pbox":

            if self.steps != other.steps:
                raise ArithmeticError(
                    "Both Pboxes must have the same number of steps")

            if method == "f":

                nleft = np.empty(self.steps)
                nright = np.empty(self.steps)

                for i in range(0, self.steps):
                    j = np.array(range(i, self.steps))
                    k = np.array(range(self.steps - 1, i - 1, -1))

                    nright[i] = np.min(self.right[j] * other.right[k])

                    jj = np.array(range(0, i + 1))
                    kk = np.array(range(i, -1, -1))

                    nleft[i] = np.max(self.left[jj] * other.left[kk])

            elif method == "p":

                nleft = self.left * other.left
                nright = self.right * other.right

            elif method == "o":

                nleft = self.left * np.flip(other.right)
                nright = self.right * np.flip(other.left)

            elif method == "i":

                nleft = []
                nright = []
                for i in self.left:
                    for j in other.left:
                        nleft.append(i * j)
                for ii in self.right:
                    for jj in other.right:
                        nright.append(ii * jj)

            nleft.sort()
            nright.sort()

            return Pbox(left=nleft, right=nright, steps=self.steps)

        else:
            try:
                # Try adding constant
                if self.shape in [
                    "uniform",
                    "normal",
                    "cauchy",
                    "triangular",
                    "skew-normal",
                ]:
                    s = self.shape
                else:
                    s = ""

                return Pbox(
                    left=self.left * other,
                    right=self.right * other,
                    shape=s,
                    mean_left=self.mean_left * other,
                    mean_right=self.mean_right * other,
                    var_left=self.var_left,
                    var_right=self.var_right,
                    steps=self.steps,
                )

            except:
                return NotImplemented

    def div(self, other, method="f"):

        if method == "o":
            method = "p"
        elif method == "p":
            method = "o"

        return self.mul(1 / other, method)

    def exp(self):
        return self._unary(function=lambda x: x.exp())

    def sqrt(self):
        return self._unary(function=lambda x: x.sqrt())

    def recip(self):
        return Pbox(
            left=1 / np.flip(self.right), right=1 / np.flip(self.left), steps=self.steps
        )

    def lt(self, other, method="f"):
        b = self.add(-other, method)
        return b.get_probability(
            0
        )  # return (self.add(-other, method)).get_probability(0)

    def le(self, other, method="f"):
        b = self.add(-other, method)
        return b.get_probability(
            0
        )  # how is the "or equal to" affecting the calculation?

    def gt(self, other, method="f"):
        self = -self
        b = self.add(other, method)
        return b.get_probability(0)  # maybe 1-prob ?

    def ge(self, other, method="f"):
        self = -self
        b = self.add(other, method)
        return b.get_probability(0)

    def min(self, other, method="f"):
        """
        Returns a new Pbox object that represents the element-wise minimum of two Pboxes.

        Parameters:
            - other: Another Pbox object or a numeric value.
            - method: Calculation method to determine the minimum. Can be one of 'f', 'p', 'o', 'i'.

        Returns:
            Pbox
        """

        if method not in ["f", "p", "o", "i"]:
            raise ArithmeticError("Calculation method unkown")

        if other.__class__.__name__ != "Pbox":
            other = Pbox(other)

        if other.__class__.__name__ == "Pbox":

            # if self.steps != other.steps:
            #     raise ArithmeticError("Both Pboxes must have the same number of steps")

            if method == "f":

                nleft = np.empty(self.steps)
                nright = np.empty(self.steps)

                for i in range(0, self.steps):
                    j = np.array(range(i, self.steps))
                    k = np.array(range(self.steps - 1, i - 1, -1))

                    nright[i] = min(list(self.right[j]) + list(other.right[k]))

                    jj = np.array(range(0, i + 1))
                    kk = np.array(range(i, -1, -1))

                    nleft[i] = min(list(self.left[jj]) + list(other.left[kk]))

            elif method == "p":

                nleft = np.minimum(self.left, other.left)
                nright = np.minimum(self.right, other.right)

            elif method == "o":

                nleft = np.minimum(self.left, np.flip(other.left))
                nright = np.minimum(self.right, np.flip(other.right))

            elif method == "i":

                nleft = []
                nright = []
                for i in self.left:
                    for j in other.left:
                        nleft.append(np.minimum(i, j))
                for ii in self.right:
                    for jj in other.right:
                        nright.append(np.minimum(ii, jj))

            nleft.sort()
            nright.sort()

            return Pbox(left=nleft, right=nright, steps=self.steps)

    def max(self, other, method="f"):

        if method not in ["f", "p", "o", "i"]:
            raise ArithmeticError("Calculation method unkown")

        if other.__class__.__name__ == "nInterval":
            other = Pbox(other, steps=self.steps)

        if other.__class__.__name__ == "Pbox":

            # if self.steps != other.steps:
            #     raise ArithmeticError("Both Pboxes must have the same number of steps")

            if method == "f":

                nleft = np.empty(self.steps)
                nright = np.empty(self.steps)

                for i in range(0, self.steps):
                    j = np.array(range(i, self.steps))
                    k = np.array(range(self.steps - 1, i - 1, -1))

                    nright[i] = max(list(self.right[j]) + list(other.right[k]))

                    jj = np.array(range(0, i + 1))
                    kk = np.array(range(i, -1, -1))

                    nleft[i] = max(list(self.left[jj]) + list(other.left[kk]))

            elif method == "p":

                nleft = np.maximum(self.left, other.left)
                nright = np.maximum(self.right, other.right)

            elif method == "o":

                nleft = np.maximum(self.left, np.flip(other.right))
                nright = np.maximum(self.right, np.flip(other.left))

            elif method == "i":

                nleft = []
                nright = []
                for i in self.left:
                    for j in other.left:
                        nleft.append(np.maximum(i, j))
                for ii in self.right:
                    for jj in other.right:
                        nright.append(np.maximum(ii, jj))

            nleft.sort()
            nright.sort()

            return Pbox(left=nleft, right=nright, steps=self.steps)

        else:
            try:
                # Try constant
                nleft = [i if i > other else other for i in self.left]
                nright = [i if i > other else other for i in self.right]

                return Pbox(left=nleft, right=nright, steps=self.steps)

            except:
                return NotImplemented

    def truncate(self, a, b, method="f"):
        """
        Equivalent to self.min(a,method).max(b,method)
        """
        return self.min(a, method=method).max(b, method=method)

# * --------------------- aggregations--------------------- *#
    def env(self, other):
        """
        .. _interval.env:

        Computes the envelope of two Pboxes.

        Parameters:
        - other: Pbox or numeric value
            The other Pbox or numeric value to compute the envelope with.

        Returns:
        - Pbox
            The envelope Pbox.

        Raises:
        - ArithmeticError: If both Pboxes have different number of steps.
        """
        # legacy bp
        # if other.__class__.__name__ == "Pbox":
        #     if self.steps != other.steps:
        #         raise ArithmeticError(
        #             "Both Pboxes must have the same number of steps")
        # else:
        #     other = Pbox(other, steps=self.steps)

        self.steps_check()
        nleft = np.minimum(self.left, other.left)
        nright = np.maximum(self.right, other.right)
        return Pbox(left=nleft, right=nright, steps=self.steps)

    def imp(self, other):
        """ Returns the imposition of self with other pbox

        note:
            - binary imposition between two pboxes only
        """
        u = []
        d = []

        assert self.steps == other.steps

        for sL, sR, oL, oR in zip(self.left, self.right, other.left, other.right):

            if max(sL, oL) > min(sR, oR):
                raise Exception("Imposition does not exist")

            u.append(max(sL, oL))
            d.append(min(sR, oR))

        return Pbox(left=u, right=d)


# * ---------------------other operations--------------------- *#

    def logicaland(self, other, method="f"):  # conjunction
        if method == "i":
            return self.mul(other, method)  # independence a * b
        elif method == "p":
            return self.min(other, method)  # perfect min(a, b)
        elif method == "o":
            # opposite max(a + b – 1, 0)
            return max(self.add(other, method) - 1, 0)
        elif method == "+":
            return self.min(other, method)  # positive env(a * b, min(a, b))
        elif method == "-":
            # negative env(max(a + b – 1, 0), a * b)
            return self.min(other, method)
        else:
            return self.env(max(0, self.add(other, method) - 1), self.min(other, method))

    def logicalor(self, other, method="f"):  # disjunction
        if method == "i":
            # independent 1 – (1 – a) * (1 – b)
            return 1 - (1 - self) * (1 - other)
        elif method == "p":
            return self.max(other, method)  # perfect max(a, b)
        elif method == "o":
            return min(self.add(other, method), 1)  # opposite min(1, a + b)
        # elif method=='+':
        #    return(env(,min(self.add(other,method),1))  # positive env(max(a, b), 1 – (1 – a) * (1 – b))
        # elif method=='-':
        #    return()  # negative env(1 – (1 – a) * (1 – b), min(1, a + b))
        else:
            return self.env(self.max(other, method), min(self.add(other, method), 1))

    def get_interval(self, *args) -> nInterval:

        if len(args) == 1:

            if args[0] == 1:
                # asking for whole pbox bounds
                return nInterval(min(self.left), max(self.right))

            p1 = (1 - args[0]) / 2
            p2 = 1 - p1

        elif len(args) == 2:

            p1 = args[0]
            p2 = args[1]

        else:
            raise Exception("Too many inputs")

        y = np.append(np.insert(np.linspace(0, 1, self.steps), 0, 0), 1)

        y1 = 0
        while y[y1] < p1:
            y1 += 1

        y2 = len(y) - 1
        while y[y2] > p2:
            y2 -= 1

        x1 = self.left[y1]
        x2 = self.right[y2]
        return nInterval(x1, x2)

    def get_probability(self, val) -> nInterval:
        p = np.append(np.insert(np.linspace(0, 1, self.steps), 0, 0), 1)

        i = 0
        while i < self.steps and self.left[i] < val:
            i += 1

        ub = p[i]

        j = 0

        while j < self.steps and self.right[j] < val:
            j += 1

        lb = p[j]

        return nInterval(lb, ub)

    def summary(self) -> str:

        return self.__repr__()

    def mean(self) -> nInterval:
        """
        Returns the mean of the pbox
        """
        return nInterval(self.mean_left, self.mean_right)

    def median(self) -> nInterval:
        """
        Returns the median of the distribution
        """
        return nInterval(np.median(self.left), np.median(self.right))

    def support(self) -> nInterval:
        return nInterval(min(self.left), max(self.right))

    def get_x(self):
        """returns the x values for plotting"""
        left = np.append(
            np.insert(self.left, 0, min(self.left)), max(self.right))
        right = np.append(
            np.insert(self.right, 0, min(self.left)), max(self.right))
        return left, right

    def get_y(self):
        """returns the y values for plotting"""
        return np.append(np.insert(np.linspace(0, 1, self.steps), 0, 0), 1)

    def straddles(self, N, endpoints=True) -> bool:
        """
        Parameters
        ----------
        N : numeric
            Number to check
        endpoints : bool
            Whether to include the endpoints within the check

        Returns
        -------
        True
            If :math:`\\mathrm{left} \\leq N \\leq \mathrm{right}` (Assuming `endpoints=True`)
        False
            Otherwise
        """
        if endpoints:
            if min(self.left) <= N and max(self.right) >= N:
                return True
        else:
            if min(self.left) < N and max(self.right) > N:
                return True

        return False

    def straddles_zero(self, endpoints=True) -> bool:
        """
        Checks whether :math:`0` is within the p-box
        """
        return self.straddles(0, endpoints)

    # * ---------------------plotting stuff---------------------#

    def show(self, figax=None, now=True, title="", x_axis_label="x", **kwargs):
        """ legacy plotting function """

        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax

        # now respects discretization
        L = self.left
        R = self.right
        steps = self.steps

        # flag: must be something wrong herein

        LL = np.concatenate((L, L, np.array([R[-1]])))
        RR = np.concatenate((np.array([L[0]]), R, R))
        ii = (
            np.concatenate(
                (np.arange(steps), np.arange(1, steps + 1), np.array([steps]))
            )
            / steps
        )
        jj = (
            np.concatenate((np.array([0]), np.arange(
                steps + 1), np.arange(1, steps)))
            / steps
        )

        ii.sort()
        jj.sort()
        LL.sort()
        RR.sort()

        if "color" in kwargs.keys():

            ax.plot(LL, ii, **kwargs)
            ax.plot(RR, jj, **kwargs)
        else:
            ax.plot(LL, ii, "r-", **kwargs)
            ax.plot(RR, jj, "k-", **kwargs)

        if title != "":
            ax.set_title(title, **kwargs)

        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(r"$\Pr(x \leq X)$")

        if now:
            plt.show()
            # fig.show() # keep the original figure open
        else:
            return fig, ax

    # @ mpl.rc_context({"text.usetex": True})
    def display(self, title="", ax=None, style="band", fill_color='lightgray', bound_colors=None, **kwargs):
        """ default plotting function """

        if ax is None:
            fig, ax = plt.subplots()

        L = self.left
        R = self.right

        # percentiles axis
        p_axis = np.linspace(0, 1, self.steps+1)
        LL_n = np.concatenate((L, np.array([R[-1]])))
        RR_n = np.concatenate((np.array([L[0]]), R))

        if bound_colors is None:
            ax.plot(LL_n, p_axis, color='g')
            ax.plot(RR_n, p_axis, color='b')
        else:
            ax.plot(LL_n, p_axis, color=bound_colors[0])
            ax.plot(RR_n, p_axis, color=bound_colors[1])

        if title is not None:
            ax.set_title(title, **kwargs)
        if style == "band":
            ax.fill_betweenx(
                y=p_axis,
                x1=LL_n,
                x2=RR_n,
                interpolate=True,
                color=fill_color,
                alpha=0.3,
            )
        elif style == "simple":
            pass
        else:
            raise ValueError("style must be either 'simple' or 'band'")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\Pr(X \leq x)$")
        return ax

    # * ---------------------conversion--------------------- *#

    def to_ds_old(self, discretisation=Params.steps):
        """convert to ds object

        note:
            - without outer approximation
        """
        p_values = np.arange(0, discretisation) / discretisation
        interval_list = [self.cuth(p_v) for p_v in p_values]
        ds = importlib.import_module("DempsterShafer")
        return ds.DempsterShafer(interval_list, np.repeat(a=(1 / discretisation), repeats=discretisation))

    def to_ds(self, discretisation=Params.steps):
        """convert to ds object"""

        _, interval_list = self.outer_approximate(discretisation)
        ds = importlib.import_module("DempsterShafer")
        return ds.DempsterShafer(interval_list, np.repeat(a=(1 / discretisation), repeats=discretisation-1))


# * ---------------------module functions--------------------- *#


def env_int(*args):
    left = min([min(i) if hasattr(i, "__iter__") else i for i in args])
    right = max([max(i) if hasattr(i, "__iter__") else i for i in args])
    return nInterval(left, right)


def left(imp):
    if isinstance(imp, nInterval) or isinstance(
        imp, Pbox
    ):  # neither "pba.pbox.Pbox" nor "pbox.Pbox" works (with or without quotemarks), even though type(b) is <class 'pba.pbox.Pbox' and isinstance(pba.norm(5,1),pba.pbox.Pbox) is True
        return imp.left
    elif hasattr(imp, "__iter__"):
        return min(imp)
    else:
        return imp


def right(imp):
    if isinstance(imp, nInterval) or isinstance(imp, Pbox):
        return imp.right
    elif hasattr(imp, "__iter__"):
        return max(imp)
    else:
        return imp


def left_list(implist, verbose=False):
    if not hasattr(implist, "__iter__"):
        return np.array(implist)

    return np.array([left(imp) for imp in implist])


def right_list(implist, verbose=False):
    if not hasattr(implist, "__iter__"):
        return np.array(implist)

    return np.array([right(imp) for imp in implist])


def interp_step(u, steps=200):
    u = np.sort(u)

    seq = np.linspace(start=0, stop=len(u) - 0.00001, num=steps, endpoint=True)
    seq = np.array([truncate(seq_val) for seq_val in seq])
    return u[seq]


def interp_cubicspline(vals, steps=200):
    vals = np.sort(vals)  # sort
    vals_steps = np.array(range(len(vals))) + 1
    vals_steps = vals_steps / len(vals_steps)

    steps = np.array(range(steps)) + 1
    steps = steps / len(steps)

    interped = interp.CubicSpline(vals_steps, vals)
    return interped(steps)


def interp_left(u, steps=200):
    p = np.array(range(len(u))) / (len(u) - 1)
    pp, x = ii(steps=steps), u
    return qleftquantiles(pp, x, p)


def interp_right(d, steps=200):
    p = np.array(range(len(d))) / (len(d) - 1)
    pp, x = jj(steps=steps), d
    return qrightquantiles(pp, x, p)


def interp_outer(x, left, steps=200):
    if left:
        return interp_left(x, steps=steps)
    else:
        return interp_right(x, steps=steps)


def interp_linear(V, steps=200):
    m = len(V) - 1

    if m == 0:
        return np.repeat(V, steps)
    if steps == 1:
        return np.array([min(V), max(V)])

    d = 1 / m
    n = round(d * steps * 200)

    if n == 0:
        c = V
    else:
        c = []
        for i in range(m):
            v = V[i]
            w = V[i + 1]
            c.extend(np.linspace(start=v, stop=w, num=n))

    u = [c[round((len(c) - 1) * (k + 0) / (steps - 1))] for k in range(steps)]

    return np.array(u)


def _interpolate(u, interpolation="linear", left=True, steps=200):
    if interpolation == "outer":
        return interp_outer(u, left, steps=steps)
    elif interpolation == "spline":
        return interp_cubicspline(u, steps=steps)
    elif interpolation == "step":
        return interp_step(u, steps=steps)
    else:
        return interp_linear(u, steps=steps)


def _sideVariance(w, mu=None):
    if not isinstance(w, np.ndarray):
        w = np.array(w)
    if mu is None:
        mu = np.mean(w)
    return max(0, np.mean((w - mu) ** 2))


def _dwMean(pbox):
    return nInterval(np.mean(pbox.right), np.mean(pbox.left))


def _dwMean(pbox):
    if np.any(np.isinf(pbox.left)) or np.any(np.isinf(pbox.right)):
        return nInterval(0, np.inf)

    if np.all(pbox.right[0] == pbox.right) and np.all(pbox.left[0] == pbox.left):
        return nInterval(0, (pbox.right[0] - pbox.left[0]) ** (2 / 4))

    vr = _sideVariance(pbox.left, np.mean(pbox.left))
    w = np.copy(pbox.left)
    n = len(pbox.left)

    for i in reversed(range(n)):
        w[i] = pbox.right[i]
        v = _sideVariance(w, np.mean(w))

        if np.isnan(vr) or np.isnan(v):
            vr = np.inf
        elif vr < v:
            vr = v

    if pbox.left[n - 1] <= pbox.right[0]:
        vl = 0.0
    else:
        x = pbox.right
        vl = _sideVariance(w, np.mean(w))

        for i in reversed(range(n)):
            w[i] = pbox.left[i]
            here = w[i]

            if 1 < i:
                for j in reversed(range(i - 1)):
                    if w[i] < w[j]:
                        w[j] = here

            v = _sideVariance(w, np.mean(w))

            if np.isnan(vl) or np.isnan(v):
                vl = 0
            elif v < vl:
                vl = v

    return nInterval(vl, vr)


def _DivByZeroCheck(bound):
    if 0 not in bound:
        return bound

    elif sum([b == 0 for b in bound]) > 1:
        # cant help
        raise DivisionByZero

    elif bound[0] == 0:
        if bound[1] > 0:
            e = 1e-3
            while abs(e) >= abs(bound[1]):
                e /= 10
            bound[0] = e
        else:
            e = -1e-3
            while abs(e) >= abs(bound[1]):
                e /= 10
            bound[0] = e

    elif bound[-1] == 0:
        if bound[-2] > 0:
            e = 1e-3
            while abs(e) >= abs(bound[-2]):
                e /= 10
            bound[-1] = e
        else:
            e = -1e-3
            while abs(e) >= abs(bound[-2]):
                e /= 10
            bound[-1] = e

    return bound


def truncate(pbox, min, max):
    return pbox.truncate(min, max)


def imposition(*args: Union[Pbox, nInterval, float, int]):
    """
    Returns the imposition of the p-boxes in *args

    Parameters
    ----------
    *args :
        Number of p-boxes or objects to be mixed

    Returns
    ----------
    Pbox
    """
    x = []
    for pbox in args:
        if pbox.__class__.__name__ != "Pbox":
            try:
                pbox = Pbox(pbox)
            except:
                raise TypeError(
                    "Unable to convert %s object (%s) to Pbox" % (
                        type(pbox), pbox)
                )
        x.append(pbox)

    p = x[0]

    for i in range(1, len(x)):
        p.imp(x[i])

    return p


# def mixture(
#     *args: Union[Pbox, nInterval, float, int],
#     weights: List[Union[float, int]] = [],
#     steps: int = Pbox.STEPS,
# ) -> Pbox:
#     # * legacy code from Nick
#     """
#     Mixes the pboxes in *args
#     Parameters
#     ----------
#     *args :
#         Number of p-boxes or objects to be mixed
#     weights:
#         Right side of box

#     Returns
#     ----------
#     Pbox
#     """
#     # TODO: IMPROVE READBILITY

#     x = []
#     for pbox in args:
#         if pbox.__class__.__name__ != "Pbox":
#             try:
#                 pbox = Pbox(pbox)
#             except:
#                 raise TypeError(
#                     "Unable to convert %s object (%s) to Pbox" % (
#                         type(pbox), pbox)
#                 )
#         x.append(pbox)

#     k = len(x)
#     if weights == []:
#         weights = [1] * k

#     # temporary hack
#     # k = 2
#     # x = [self, x]
#     # w = [1,1]

#     if k != len(weights):
#         return "Need same number of weights as arguments for mixture"
#     weights = [i / sum(weights) for i in weights]  # w = w / sum(w)
#     u = []
#     d = []
#     n = []
#     ml = []
#     mh = []
#     m = []
#     vl = []
#     vh = []
#     v = []
#     for i in range(k):
#         u = u + list(x[i].left)
#         d = np.append(d, x[i].right)
#         n = (
#             n + [weights[i] / x[i].steps] * x[i].steps
#         )  # w[i]*rep(1/x[i].steps,x[i].steps))

#         # mu = mean(x[i])
#         # ml = ml + [mu.left()]
#         # mh = mh + [mu.right()]
#         # m = m + [mu]               # don't need?
#         # sigma2 = var(x[[i]])  ### !!!! shouldn't be the sample variance, but the population variance
#         # vl = vl + [sigma2.left()]
#         # vh = vh + [sigma2.right()]
#         # v = v + [sigma2]

#         ML = x[i].mean_left
#         MR = x[i].mean_right
#         VL = x[i].var_left
#         VR = x[i].var_right
#         m = m + [nInterval(ML, MR)]
#         v = v + [nInterval(VL, VR)]
#         ml = ml + [ML]
#         mh = mh + [MR]
#         vl = vl + [VL]
#         vh = vh + [VR]

#     n = [_ / sum(n) for _ in n]  # n = n / sum(n)
#     su = sorted(u)
#     su = [su[0]] + su
#     pu = [0] + list(
#         np.cumsum([n[i] for i in np.argsort(u)])
#     )  # pu = c(0,cumsum(n[order(u)]))
#     sd = sorted(d)
#     sd = sd + [sd[-1]]
#     pd = list(np.cumsum([n[i] for i in np.argsort(d)])) + [
#         1
#     ]  # pd = c(cumsum(n[order(d)]),1)
#     u = []
#     d = []
#     j = len(pu) - 1
#     for p in reversed(
#         np.arange(steps) / steps
#     ):  # ii = np.arange(steps))/steps  #    ii = 0: (Pbox$steps-1) / Pbox$steps
#         while p < pu[j]:
#             j = j - 1  # repeat {if (pu[j] <= p) break; j = j - 1}
#         u = [su[j]] + u
#     j = 0
#     for p in (
#         np.arange(steps) + 1
#     ) / steps:  # jj = (np.arange(steps)+1)/steps #  jj =  1: Pbox$steps / Pbox$steps
#         while pd[j] < p:
#             j = j + 1  # repeat {if (p <= pu[j]) break; j = j + 1}
#         d = d + [sd[j]]
#     mu = nInterval(
#         np.sum([W * M for M, W in zip(weights, ml)]),
#         np.sum([W * M for M, W in zip(weights, mh)]),
#     )
#     s2 = 0
#     for i in range(k):
#         s2 = s2 + weights[i] * (v[i] + m[i] ** 2)
#     s2 = s2 - mu**2

#     return Pbox(
#         np.array(u),
#         np.array(d),
#         mean_left=mu.left,
#         mean_right=mu.right,
#         var_left=s2.left,
#         var_right=s2.right,
#         steps=steps,
#     )
