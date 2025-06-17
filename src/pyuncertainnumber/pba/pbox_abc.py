import numpy as np
from abc import ABC, abstractmethod
from .params import Params
import matplotlib.pyplot as plt
from .intervals.number import Interval as I
from numbers import Number
import operator
import itertools
from .utils import condensation, smooth_condensation, find_nearest, is_increasing
import logging

# Configure the logging system with a simple format
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


def get_var_from_ecdf(q, p):
    """leslie implementation

    example:
        # Given ECDF data an example
        # q = [1, 2, 3, 4]
        # p = [0.25, 0.5, 0.75, 1.0]
    """

    # Step 1: Recover PMF
    pmf = [p[0]] + [p[i] - p[i - 1] for i in range(1, len(p))]

    # Step 2: Compute Mean
    mean = sum(x * p for x, p in zip(q, pmf))

    # Step 3: Compute Variance
    variance = sum(p * (x - mean) ** 2 for x, p in zip(q, pmf))
    return mean, variance


def bound_steps_check(bound):
    # condensation needed
    if len(bound) > Params.steps:
        bound = condensation(bound, Params.steps)
    elif len(bound) < Params.steps:
        # 'next' kind interpolation needed
        from .constructors import interpolate_p

        p_lo, bound = interpolate_p(
            p=np.linspace(Params.p_lboundary, Params.p_hboundary, len(bound)), q=bound
        )
    return bound


# * --------------------- constructors ---------------------*#
def pbox_from_extredists(rvs, shape="beta", extre_bound_params=None):
    """transform into pbox object from extreme bounds parameterised by `sps.dist`

    args:
        rvs (list): list of scipy.stats.rv_continuous objects"""

    # x_sup
    bounds = [rv.ppf(Params.p_values) for rv in rvs]

    if bounds[0][-1] > bounds[1][-1]:
        # swap left and right bounds
        bounds[0], bounds[1] = bounds[1], bounds[0]

    if extre_bound_params is not None:
        print(extre_bound_params)

    return Staircase(
        left=bounds[0],
        right=bounds[1],
        shape=shape,
    )


class Pbox(ABC):
    """a base class for Pbox"""

    def __init__(
        self,
        left: np.ndarray | list,
        right: np.ndarray | list,
        steps=Params.steps,
        mean=None,
        var=None,
        p_values=None,
    ):
        self.left = np.array(left)
        self.right = np.array(right)
        self.steps = steps
        self.mean = mean
        self.var = var
        # we force the steps but allow the p_values to be flexible
        self._pvalues = p_values if p_values is not None else Params.p_values
        self.post_init_check()

    # * --------------------- setup ---------------------*#

    @abstractmethod
    def _init_moments(self):
        pass

    def _init_range(self):
        self._range = I(min(self.left), max(self.right))

    def post_init_check(self):

        self.steps_check()

        if (not is_increasing(self.left)) or (not is_increasing(self.right)):
            raise Exception("Left and right arrays must be increasing")

        # pass along moments information
        if (self.mean is None) and (self.var is None):
            self._init_moments()

        self._init_range()

    def steps_check(self):

        assert len(self.left) == len(
            self.right
        ), "Length of lower and upper bounds is not consistent"

    @property
    def p_values(self):
        return self._pvalues

    @property
    def range(self):
        return self._range

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        self._left = bound_steps_check(value)
        self.steps = len(self._left)

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        self._right = bound_steps_check(value)
        self.steps = len(self._right)

    @property
    def lo(self):
        """Returns the left-most value in the interval"""
        return self.left[0]

    @property
    def hi(self):
        """Returns the right-most value in the interval"""
        return self.right[-1]

    @property
    def support(self):
        return self._range

    @property
    def median(self):
        return I(np.median(self.left), np.median(self.right))

    @property
    def naked_value(self):
        return np.round(self.mean.mid, 3)

    @property
    def area_metric(self):
        return np.trapezoid(y=self._pvalues, x=self.left) - np.trapezoid(
            y=self._pvalues, x=self.right
        )

    # * --------------------- operators ---------------------*#

    def __iter__(self):
        return iter(self.to_interval())

    # * --------------------- functions ---------------------*#
    def to_interval(self):
        from .intervals.number import Interval as I

        return I(lo=self.left, hi=self.right)


class Staircase(Pbox):
    """distribution free p-box"""

    # TODO with the automatic interpolation, there is no point still aving the `p_values` as a parameter
    def __init__(
        self,
        left,
        right,
        steps=200,
        mean=None,
        var=None,
        p_values=None,
    ):
        super().__init__(left, right, steps, mean, var, p_values)

    def _init_moments(self):
        """initialised `mean`, `var` and `range` bounds"""

        #! should we compute mean if it is a Cauchy, var if it's a t distribution?
        #! we assume that two extreme bounds are valid CDFs
        self.mean_lo, self.var_lo = get_var_from_ecdf(self.left, self._pvalues)
        self.mean_hi, self.var_hi = get_var_from_ecdf(self.right, self._pvalues)
        self.mean = I(self.mean_lo, self.mean_hi)
        # TODO tmp solution for computing var for pbox
        try:
            self.var = I(self.var_lo, self.var_hi)
        except:
            self.var = I(666, 666)

    def __repr__(self):
        def format_interval(interval):
            try:
                return f"[{interval.lo:.3f}, {interval.hi:.3f}]"
            except Exception:
                return str(interval)

        mean_text = format_interval(self.mean)
        var_text = format_interval(self.var)
        range_text = format_interval(self._range)

        return f"Pbox ~ (range={range_text}, mean={mean_text}, var={var_text})"

    def plot(
        self,
        title=None,
        ax=None,
        style="box",
        fill_color="lightgray",
        bound_colors=None,
        nuance="step",
        alpha=0.3,
        **kwargs,
    ):
        """default plotting function

        args:
            style (str): 'box' or 'simple'
        """
        from .utils import CustomEdgeRectHandler

        if ax is None:
            fig, ax = plt.subplots()

        p_axis = self._pvalues if self._pvalues is not None else Params.p_values
        plot_bound_colors = bound_colors if bound_colors is not None else ["g", "b"]

        def display_box(nuance, label=None):
            """display two F curves plus the top-bottom horizontal lines"""

            if nuance == "step":
                step_kwargs = {
                    "c": plot_bound_colors[0],
                    "where": "post",
                }

                if label is not None:
                    step_kwargs["label"] = label

                # Make the plot
                (line,) = ax.step(self.left, p_axis, **step_kwargs)
                ax.step(self.right, p_axis, c=plot_bound_colors[1], where="post")
                ax.plot([self.left[0], self.right[0]], [0, 0], c=plot_bound_colors[1])
                ax.plot([self.left[-1], self.right[-1]], [1, 1], c=plot_bound_colors[0])
            elif nuance == "curve":
                smooth_curve_kwargs = {
                    "c": plot_bound_colors[0],
                }

                if label is not None:
                    smooth_curve_kwargs["label"] = label

                (line,) = ax.plot(self.left, p_axis, **smooth_curve_kwargs)
                ax.plot(self.right, p_axis, c=plot_bound_colors[1])
                ax.plot([self.left[0], self.right[0]], [0, 0], c=plot_bound_colors[1])
                ax.plot([self.left[-1], self.right[-1]], [1, 1], c=plot_bound_colors[0])
            else:
                raise ValueError("nuance must be either 'step' or 'curve'")
            if label is not None:
                ax.legend(handler_map={line: CustomEdgeRectHandler()})  # regular use

        if title is not None:
            ax.set_title(title)
        if style == "box":
            ax.fill_betweenx(
                y=p_axis,
                x1=self.left,
                x2=self.right,
                interpolate=True,
                color=fill_color,
                alpha=alpha,
                **kwargs,
            )
            display_box(nuance, label=None)
            if "label" in kwargs:
                ax.legend(loc="best")
        elif style == "simple":
            display_box(nuance, label=kwargs["label"] if "label" in kwargs else None)
        else:
            raise ValueError("style must be either 'simple' or 'box'")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\Pr(X \leq x)$")
        return ax

    def plot_outside_legend(
        self,
        title=None,
        ax=None,
        style="box",
        fill_color="lightgray",
        bound_colors=None,
        nuance="step",
        alpha=0.3,
        **kwargs,
    ):
        """a specific variant of `plot()` which is used for scipy proceeding only.

        args:
            style (str): 'box' or 'simple'
        """
        from .utils import CustomEdgeRectHandler

        if ax is None:
            fig, ax = plt.subplots()

        p_axis = self._pvalues if self._pvalues is not None else Params.p_values
        plot_bound_colors = bound_colors if bound_colors is not None else ["g", "b"]

        def display_box(nuance, label=None):
            """display two F curves plus the top-bottom horizontal lines"""

            if nuance == "step":
                step_kwargs = {
                    "c": plot_bound_colors[0],
                    "where": "post",
                }

                if label is not None:
                    step_kwargs["label"] = label

                # Make the plot
                (line,) = ax.step(self.left, p_axis, **step_kwargs)
                ax.step(self.right, p_axis, c=plot_bound_colors[1], where="post")
                ax.plot([self.left[0], self.right[0]], [0, 0], c=plot_bound_colors[1])
                ax.plot([self.left[-1], self.right[-1]], [1, 1], c=plot_bound_colors[0])
            elif nuance == "curve":
                smooth_curve_kwargs = {
                    "c": plot_bound_colors[0],
                }

                if label is not None:
                    smooth_curve_kwargs["label"] = label

                (line,) = ax.plot(self.left, p_axis, **smooth_curve_kwargs)
                ax.plot(self.right, p_axis, c=plot_bound_colors[1])
                ax.plot([self.left[0], self.right[0]], [0, 0], c=plot_bound_colors[1])
                ax.plot([self.left[-1], self.right[-1]], [1, 1], c=plot_bound_colors[0])
            else:
                raise ValueError("nuance must be either 'step' or 'curve'")
            if label is not None:
                # ax.legend(handler_map={line: CustomEdgeRectHandler()})  # regular use
                # Put a legend to the right of the current axis
                ax.legend(
                    handler_map={line: CustomEdgeRectHandler()},
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                )  # onetime use

        if title is not None:
            ax.set_title(title)
        if style == "box":
            ax.fill_betweenx(
                y=p_axis,
                x1=self.left,
                x2=self.right,
                interpolate=True,
                color=fill_color,
                alpha=alpha,
                **kwargs,
            )
            display_box(nuance, label=None)
            if "label" in kwargs:
                ax.legend(loc="best")
        elif style == "simple":
            display_box(nuance, label=kwargs["label"] if "label" in kwargs else None)
        else:
            raise ValueError("style must be either 'simple' or 'box'")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\Pr(X \leq x)$")
        return ax

    def display(self, *args, **kwargs):
        self.plot(*args, **kwargs)
        plt.show()

    def plot_probability_bound(self, x: float, ax=None, **kwargs):
        """plot the probability bound at a certain quantile x

        note:
            - a vertical line
        """

        if ax is None:
            fig, ax = plt.subplots()

        p_lo = self.cdf(x).lo
        p_hi = self.cdf(x).hi
        self.plot(ax=ax, **kwargs)

        ax.plot(
            [x, x],
            [p_lo, p_hi],
            c="r",
            label="probability bound",
            zorder=50,
        )
        ax.scatter(x, p_lo, c="r", marker="^", zorder=50)
        ax.scatter(x, p_hi, c="r", marker="v", zorder=50)
        return ax

    # * --------------------- constructors ---------------------*#
    @classmethod
    def from_CDFbundle(cls, a, b):
        """pbox from two emipirical CDF bundle

        args:
            - a : CDF bundle of lower extreme F;
            - b : CDF bundle of upper extreme F;
        """
        from .constructors import interpolate_p
        from .utils import extend_ecdf

        a = extend_ecdf(a)
        b = extend_ecdf(b)

        p_lo, q_lo = interpolate_p(a.probabilities, a.quantiles)
        p_hi, q_hi = interpolate_p(b.probabilities, b.quantiles)
        return cls(left=q_lo, right=q_hi, p_values=p_lo)

    # * --------------------- operators ---------------------*#

    def __neg__(self):
        return Staircase(
            left=sorted(-np.flip(self.right)),
            right=sorted(-np.flip(self.left)),
            mean=-self.mean,
            var=self.var,
        )

    def __add__(self, other):
        return self.add(other, dependency="f")

    def __radd__(self, other):
        return self.add(other, dependency="f")

    def __sub__(self, other):
        return self.sub(other, dependency="f")

    def __rsub__(self, other):
        self = -self
        return self.add(other, dependency="f")

    def __mul__(self, other):
        return self.mul(other, dependency="f")

    def __rmul__(self, other):
        return self.mul(other, dependency="f")

    def __truediv__(self, other):

        return self.div(other, dependency="f")

    def __rtruediv__(self, other):

        try:
            return other * self.recip()
        except:
            return NotImplemented

    def __pow__(self, other):
        return self.pow(other, dependency="f")

    def __rpow__(self, other: Number):
        # if not hasattr(other, "__iter__"):
        #     other = np.array((other))
        from functools import partial

        bar = partial(np.power, other)
        return self._unary_template(bar)

    # * --------------------- methods ---------------------*#

    def cdf(self, x: np.ndarray):
        """get the bounds on the cdf w.r.t x value

        args:
            x (array-like): x values
        """
        lo_ind = find_nearest(self.right, x)
        hi_ind = find_nearest(self.left, x)
        return I(lo=Params.p_values[lo_ind], hi=Params.p_values[hi_ind])

    def alpha_cut(self, alpha=0.5):
        """get the bounds on the quantile at any particular probability level

        args:
            alpha (array-like): probability levels
        """
        ind = find_nearest(Params.p_values, alpha)
        return I(lo=self.left[ind], hi=self.right[ind])

    def sample(self, n_sam):
        from scipy.stats import qmc

        alpha = np.squeeze(qmc.LatinHypercube(d=1).random(n=n_sam))
        return self.alpha_cut(alpha)

    def discretise(self, n=None):
        from .utils import equi_selection

        if (n is None) or (n == Params.steps):
            return I(lo=self.left, hi=self.right)
        else:
            l, r = equi_selection(self.left, n), equi_selection(self.right, n)
            return I(lo=l, hi=r)

    def outer_approximate(self, n=None):
        """outer approximation of a p-box

        args:
            - n: number of steps to be used in the approximation
        note:
            - `the_interval_list` will have length one less than that of `p_values` (i.e. 100 and 99)
        """

        from .intervals.number import Interval as I

        if n is not None:
            p_values = np.arange(0, n) / n
        else:
            p_values = self._pvalues

        p_leftend = p_values[0:-1]
        p_rightend = p_values[1:]

        q_l = [self.alpha_cut(p).left for p in p_leftend]
        q_r = [self.alpha_cut(p).right for p in p_rightend]

        interval_vec = I(lo=q_l, hi=q_r)
        return p_values, interval_vec

    def area_metric(self):
        return np.trapezoid(y=self.left, x=self._pvalues) - np.trapezoid(
            y=self.right, x=self._pvalues
        )

    def truncate(self, a, b, method="f"):
        """Equivalent to self.min(a,method).max(b,method)"""
        return self.min(a, method=method).max(b, method=method)

    def min(self, other, method="f"):
        """Returns a new Pbox object that represents the element-wise minimum of two Pboxes.

        args:
            - other: Another Pbox object or a numeric value.
            - method: Calculation method to determine the minimum. Can be one of 'f', 'p', 'o', 'i'.

        returns:
            Pbox
        """

        other = convert_pbox(other)
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

        return Staircase(left=nleft, right=nright)

    def max(self, other, method="f"):

        other = convert_pbox(other)
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

        return Staircase(left=nleft, right=nright)

    # * --------------------- aggregations--------------------- *#
    def env(self, other):
        """computes the envelope of two Pboxes.

        args:
            other (Pbox)

        returns:
            - Pbox
        """

        nleft = np.minimum(self.left, other.left)
        nright = np.maximum(self.right, other.right)
        return Staircase(left=nleft, right=nright, steps=self.steps)

    def imp(self, other):
        """Returns the imposition of self with other pbox

        note:
            - binary imposition between two pboxes only
        """
        u = []
        d = []
        for sL, sR, oL, oR in zip(self.left, self.right, other.left, other.right):
            if max(sL, oL) > min(sR, oR):
                raise Exception("Imposition does not exist")
            u.append(max(sL, oL))
            d.append(min(sR, oR))
        return Staircase(left=u, right=d)

    # * ---------------------unary operations--------------------- *#

    def _unary_template(self, f):
        l, r = f(self.left), f(self.right)
        return Staircase(left=l, right=r)

    def exp(self):
        return self._unary_template(np.exp)

    def sqrt(self):
        return self._unary_template(np.sqrt)

    def recip(self):
        return Staircase(left=1 / np.flip(self.right), right=1 / np.flip(self.left))

    def log(self):
        """natural logarithm of the pbox

        note:
            - the pbox must be positive
        """
        if self.lo <= 0:
            raise ValueError("Logarithm is not defined for non-positive values")
        return self._unary_template(np.log)

    def sin(self):
        from .intervals.methods import sin

        itvls = sin(self.to_interval())
        return simple_stacking(itvls)

    def cos(self):
        from .intervals.methods import cos

        itvls = cos(self.to_interval())
        return simple_stacking(itvls)

    def tanh(self):
        from .intervals.methods import tanh

        itvls = tanh(self.to_interval())
        return simple_stacking(itvls)

    # * ---------------------binary operations--------------------- *#

    def add(self, other, dependency="f"):
        if isinstance(other, Number):
            return pbox_number_ops(self, other, operator.add)
        if is_un(other):
            other = convert_pbox(other)
        match dependency:
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
                for l in itertools.product(self.left, other.left):
                    nleft.append(operator.add(*l))
                for r in itertools.product(self.right, other.right):
                    nright.append(operator.add(*r))
        nleft.sort()
        nright.sort()
        return Staircase(left=nleft, right=nright)

    def sub(self, other, dependency="f"):

        if dependency == "o":
            dependency = "p"
        elif dependency == "p":
            dependency = "o"

        return self.add(-other, dependency)

    def mul(self, other, dependency="f"):
        """Multiplication of uncertain numbers with the defined dependency dependency"""

        if isinstance(other, Number):
            return pbox_number_ops(self, other, operator.mul)
        if is_un(other):
            other = convert_pbox(other)

        match dependency:
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
        return Staircase(left=nleft, right=nright)

    def div(self, other, dependency="f"):

        if dependency == "o":
            dependency = "p"
        elif dependency == "p":
            dependency = "o"

        return self.mul(1 / other, dependency)

    def pow(self, other, dependency="f"):

        if isinstance(other, Number):
            return pbox_number_ops(self, other, operator.pow)
        if is_un(other):
            other = convert_pbox(other)

        match dependency:
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
        return Staircase(left=nleft, right=nright)


class Leaf(Staircase):
    """parametric pbox"""

    def __init__(
        self,
        left=None,
        right=None,
        steps=200,
        mean=None,
        var=None,
        dist_params=None,
        shape=None,
    ):
        super().__init__(left, right, steps, mean, var)
        self.shape = shape
        self.dist_params = dist_params

    def _init_moments_range(self):
        print("not decided yet")

    def __repr__(self):
        base_repr = super().__repr__().rstrip(")")  # remove trailing ')'
        return f"{base_repr}, shape={self.shape}"

    def sample(self, n_sam):
        """sample from a parametric pbox or distribution"""

        s_i = super().sample(n_sam)
        if np.all(s_i.lo == s_i.hi):
            logging.info("samples generated from a precise distribution")
            return s_i.lo
        else:
            return s_i


class Cbox(Pbox):
    def __init__(self, left, right, steps=200):
        super().__init__(left, right, steps)


# * --------------------- module functions ---------------------*#


def is_un(un):
    """if the `un` is modelled by accepted constructs"""

    from .intervals.number import Interval
    from .dss import DempsterShafer
    from .distributions import Distribution

    return isinstance(un, Pbox | Interval | DempsterShafer | Distribution)


def convert_pbox(un):
    """transform the input un into a Pbox object

    note:
        - theorically 'un' can be {Interval, DempsterShafer, Distribution, float, int}
    """

    from .pbox_abc import Pbox
    from .dss import DempsterShafer
    from .distributions import Distribution
    from .intervals.number import Interval as I

    if isinstance(un, Pbox):
        return un
    elif isinstance(un, I):
        return un.to_pbox()
        # return Staircase(
        #     left=np.repeat(un.lo, Params.steps),
        #     right=np.repeat(un.hi, Params.steps),
        #     mean=un,
        #     var=I(0, (un.hi - un.lo) * (un.hi - un.lo) / 4),
        # )
    elif isinstance(un, Pbox):
        return un
    elif isinstance(un, Distribution):
        return un.to_pbox()
    elif isinstance(un, DempsterShafer):
        return un.to_pbox()
    else:
        raise TypeError(f"Unable to convert {type(un)} object to Pbox")


def pbox_number_ops(pbox: Staircase | Leaf, n: float | int, f: callable):
    # TODO: ask Scott. pbox sqrt operaton how to do?
    """blueprint for arithmetic between pbox and real numbers"""
    l = f(pbox.left, n)
    r = f(pbox.right, n)
    l = sorted(l)
    r = sorted(r)
    try:
        new_mean = f(pbox.mean, n)
    except:
        new_mean = None
    return Staircase(left=l, right=r, mean=new_mean, var=pbox.var)

    # Staircase(left=pbox.left + n, right=pbox.right + n)


def truncate(pbox, min, max):
    return pbox.truncate(min, max)


# * --------------------- unary functions ---------------------*#
def sin():
    pass


def cos():
    pass


def tanh():
    pass


def exp():
    pass


def log():
    pass


def sqrt():
    pass


# * --------------------- utility functions tmp ---------------------*#
def simple_stacking(itvls):
    """simple version of stacking vector Interval objects into pbox

    args:
        itvls (Interval): a vector Interval object to be stacked

    note:
        - only meant for quick use during development
        - see `stacking` function for production use
    """
    from .ecdf import get_ecdf, eCDF_bundle

    q1, p1 = get_ecdf(itvls.lo)
    q2, p2 = get_ecdf(itvls.hi)

    cdf1 = eCDF_bundle(q1, p1)
    cdf2 = eCDF_bundle(q2, p2)
    return Staircase.from_CDFbundle(cdf1, cdf2)
