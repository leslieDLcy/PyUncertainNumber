from .operation import convert
from decimal import DivisionByZero
from typing import Self
from warnings import *
import numpy as np
from matplotlib import pyplot as plt
from .interval import Interval as nInterval
from .utils import (
    find_nearest,
    check_increasing,
    NotIncreasingError,
    _interval_list_to_array,
)
from .params import Params
import importlib

__all__ = [
    "Pbox",
    "truncate",
    "NotIncreasingError",
]


class Pbox:
    """
    A probability distribution is a mathematical function that gives the probabilities of occurrence for diﬀerent possible values of a variable. Probability boxes (p-boxes) represent interval bounds on probability distributions. The left and right quantiles are each stored as a NumPy array containing the percent point function (the inverse of the cumulative distribution function) for `steps` evenly spaced values between 0 and 1. P-boxes can be defined using all the probability distributions that are available through SciPy's statistics library. Naturally, precis probability distributions can be defined by defining a p-box with precise inputs. This means that within probability bounds analysis probability distributions are considered a special case of a p-box with zero width. Distribution-free p-boxes can also be generated when the underlying distribution is unknown but parameters such as the mean, variance or minimum/maximum bounds are known. Such p-boxes make no assumption about the shape of the distribution and instead return bounds expressing all possible distributions that are valid given the known information. Such p-boxes can be constructed making use of Chebyshev, Markov and Cantelli inequalities from probability theory.
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
            """initialisation logic"""

            if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                if len(left) != len(right):
                    raise Exception("Left and right arrays must be the same length")
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
                raise NotIncreasingError("Left and right arrays must be increasing")

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

    # * ---------------------Basic arithmetics---------------------*#

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

    @property
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
        """get the bounds on the cumulative probability associated with any x-value"""
        lo_ind = find_nearest(self.right, x)
        hi_ind = find_nearest(self.left, x)
        return nInterval(Params.p_values[lo_ind], Params.p_values[hi_ind])

    def cuth(self, p=0.5):
        """get the bounds on the quantile at any particular probability level"""
        # TODO have a conservative cut.
        ind = find_nearest(Params.p_values, p)
        return nInterval(self.left[ind], self.right[ind])

    def outer_approximate(self, n=100):
        """outer approximation of a p-box

        note:
            - `the_interval_list` will have length one less than that of `p_values` (i.e. 100 and 99)
        """
        p_values = np.arange(0, n) / n
        p_leftend = p_values[0:-1]
        p_rightend = p_values[1:]

        q_l = [self.cuth(p).left for p in p_leftend]
        q_r = [self.cuth(p).right for p in p_rightend]

        # get the interval list
        # TODO streamline below the interval list into Marco interval vector
        the_interval_list = [(l, r) for l, r in zip(q_l, q_r)]
        return p_values, the_interval_list

    # * ---------------------unary operations--------------------- *#
    ##### the top-level functions for unary operations #####

    def _unary(self, *args, function=lambda x: x):
        """for monotonic unary functions only"""

        ints = [function(nInterval(l, r), *args) for l, r in zip(self.left, self.right)]
        return Pbox(
            left=np.array([i.left for i in ints]),
            right=np.array([i.right for i in ints]),
        )

    def exp(self):
        return self._unary(function=lambda x: x.exp())

    def sqrt(self):
        return self._unary(function=lambda x: x.sqrt())

    def recip(self):
        return Pbox(
            left=1 / np.flip(self.right), right=1 / np.flip(self.left), steps=self.steps
        )

    # * ---------------------binary operations--------------------- *#
    @staticmethod
    def check_dependency(method):
        if method not in ["f", "p", "o", "i"]:
            raise ArithmeticError("dependency not registered")

    def constant_shape_check(self):
        """a helper drop in for define binary ops"""
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
        return s

    def steps_check(self, other):
        if self.steps != other.steps:
            raise ArithmeticError("Both Pboxes must have the same number of steps")

    def add(self, other: Self | nInterval | float | int, method="f") -> Self:
        """addtion of uncertain numbers with the defined dependency method"""

        self.check_dependency(method)
        if isinstance(other, (float, int)):
            # case with constant
            try:
                s = self.constant_shape_check()
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

        else:
            other = convert(other)
            self.steps_check(other)

            match method:
                case "f":
                    nleft = np.empty(self.steps)
                    nright = np.empty(self.steps)
                    for i in range(0, self.steps):
                        j = np.array(range(i, self.steps))
                        k = np.array(range(self.steps - 1, i - 1, -1))
                        nright[i] = np.min(self.right[j] + other.right[k])
                        jj = np.array(range(0, i + 1))
                        kk = np.array(range(i, -1, -1))
                        nleft[i] = np.max(self.left[jj] + other.left[kk])
                case "p":
                    nleft = self.left + other.left
                    nright = self.right + other.right
                case "o":
                    nleft = self.left + np.flip(other.right)
                    nright = self.right + np.flip(other.left)
                case "i":
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

    def pow(self, other: Self | nInterval | float | int, method="f") -> Self:
        """Raises a p-box to the power of other using the defined dependency method"""

        self.check_dependency(method)
        if isinstance(other, (float, int)):
            # case with constant
            try:
                s = self.constant_shape_check()
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
        else:
            other = convert(other)
            self.steps_check(other)

            match method:
                case "f":
                    nleft = np.empty(self.steps)
                    nright = np.empty(self.steps)

                    for i in range(0, self.steps):
                        j = np.array(range(i, self.steps))
                        k = np.array(range(self.steps - 1, i - 1, -1))

                        nright[i] = np.min(self.right[j] ** other.right[k])

                        jj = np.array(range(0, i + 1))
                        kk = np.array(range(i, -1, -1))

                        nleft[i] = np.max(self.left[jj] ** other.left[kk])
                case "p":

                    nleft = self.left**other.left
                    nright = self.right**other.right
                case "o":

                    nleft = self.left ** np.flip(other.right)
                    nright = self.right ** np.flip(other.left)
                case "i":

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

    def sub(self, other, method="f"):

        if method == "o":
            method = "p"
        elif method == "p":
            method = "o"

        return self.add(-other, method)

    def mul(self, other, method="f"):
        """Multiplication of uncertain numbers with the defined dependency method"""

        self.check_dependency(method)
        if isinstance(other, (float, int)):
            # case with constant
            try:
                s = self.constant_shape_check()
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
        else:
            other = convert(other)
            self.steps_check(other)

            match method:
                case "f":
                    nleft = np.empty(self.steps)
                    nright = np.empty(self.steps)

                    for i in range(0, self.steps):
                        j = np.array(range(i, self.steps))
                        k = np.array(range(self.steps - 1, i - 1, -1))

                        nright[i] = np.min(self.right[j] * other.right[k])

                        jj = np.array(range(0, i + 1))
                        kk = np.array(range(i, -1, -1))

                        nleft[i] = np.max(self.left[jj] * other.left[kk])
                case "p":
                    nleft = self.left * other.left
                    nright = self.right * other.right
                case "o":
                    nleft = self.left * np.flip(other.right)
                    nright = self.right * np.flip(other.left)
                case "i":
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

    def div(self, other, method="f"):

        if method == "o":
            method = "p"
        elif method == "p":
            method = "o"

        return self.mul(1 / other, method)

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
        """Returns a new Pbox object that represents the element-wise minimum of two Pboxes.

        args:
            - other: Another Pbox object or a numeric value.
            - method: Calculation method to determine the minimum. Can be one of 'f', 'p', 'o', 'i'.

        returns:
            Pbox
        """

        self.check_dependency(method)
        other = convert(other)
        self.steps_check(other)
        match method:
            case "f":

                nleft = np.empty(self.steps)
                nright = np.empty(self.steps)

                for i in range(0, self.steps):
                    j = np.array(range(i, self.steps))
                    k = np.array(range(self.steps - 1, i - 1, -1))

                    nright[i] = min(list(self.right[j]) + list(other.right[k]))

                    jj = np.array(range(0, i + 1))
                    kk = np.array(range(i, -1, -1))

                    nleft[i] = min(list(self.left[jj]) + list(other.left[kk]))
            case "p":
                nleft = np.minimum(self.left, other.left)
                nright = np.minimum(self.right, other.right)
            case "o":
                nleft = np.minimum(self.left, np.flip(other.left))
                nright = np.minimum(self.right, np.flip(other.right))
            case "i":
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

        self.check_dependency(method)
        other = convert(other)
        self.steps_check(other)
        match method:
            case "f":
                nleft = np.empty(self.steps)
                nright = np.empty(self.steps)
                for i in range(0, self.steps):
                    j = np.array(range(i, self.steps))
                    k = np.array(range(self.steps - 1, i - 1, -1))
                    nright[i] = max(list(self.right[j]) + list(other.right[k]))
                    jj = np.array(range(0, i + 1))
                    kk = np.array(range(i, -1, -1))
                    nleft[i] = max(list(self.left[jj]) + list(other.left[kk]))
            case "p":
                nleft = np.maximum(self.left, other.left)
                nright = np.maximum(self.right, other.right)
            case "o":
                nleft = np.maximum(self.left, np.flip(other.right))
                nright = np.maximum(self.right, np.flip(other.left))
            case "i":
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

        # else:
        #     try:
        #         # Try constant
        #         nleft = [i if i > other else other for i in self.left]
        #         nright = [i if i > other else other for i in self.right]

        #         return Pbox(left=nleft, right=nright, steps=self.steps)

        #     except:
        #         return NotImplemented

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

        self.steps_check(other)
        nleft = np.minimum(self.left, other.left)
        nright = np.maximum(self.right, other.right)
        return Pbox(left=nleft, right=nright, steps=self.steps)

    def imp(self, other):
        """Returns the imposition of self with other pbox

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
            return self.env(
                max(0, self.add(other, method) - 1), self.min(other, method)
            )

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
        left = np.append(np.insert(self.left, 0, min(self.left)), max(self.right))
        right = np.append(np.insert(self.right, 0, min(self.left)), max(self.right))
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
        """legacy plotting function"""

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
            np.concatenate((np.array([0]), np.arange(steps + 1), np.arange(1, steps)))
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

    def display(
        self,
        title="",
        ax=None,
        style="band",
        fill_color="lightgray",
        bound_colors=None,
        **kwargs,
    ):
        """default plotting function"""

        if ax is None:
            fig, ax = plt.subplots()

        L = self.left
        R = self.right

        # percentiles axis
        p_axis = np.linspace(0, 1, self.steps + 1)
        LL_n = np.concatenate((L, np.array([R[-1]])))
        RR_n = np.concatenate((np.array([L[0]]), R))

        if bound_colors is None:
            ax.plot(LL_n, p_axis, color="g")
            ax.plot(RR_n, p_axis, color="b")
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
        return ds.DempsterShafer(
            interval_list, np.repeat(a=(1 / discretisation), repeats=discretisation)
        )

    def to_ds(self, discretisation=Params.steps):
        """convert to ds object"""

        _, interval_list = self.outer_approximate(discretisation)
        ds = importlib.import_module("DempsterShafer")
        return ds.DempsterShafer(
            interval_list, np.repeat(a=(1 / discretisation), repeats=discretisation - 1)
        )


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
