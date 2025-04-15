import numpy as np
from abc import ABC, abstractmethod
from .pbox_base import _sideVariance
from .utils import is_increasing
from .params import Params
import matplotlib.pyplot as plt
from .intervals.number import Interval as I
from numbers import Number
import operator
import itertools
from .utils import condensation, smooth_condensation


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


class Box(ABC):
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

    def _init_range(self):
        self._range = I(min(self.left), max(self.right))

    # * --------------------- setup ---------------------*#

    @abstractmethod
    def _init_moments(self):
        pass

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

        if len(self.left) > self.steps:
            self.left, self.right = condensation([self.left, self.right], self.steps)
        elif len(self.left) < self.steps:
            # 'next' kind interpolation needed
            from .constructors import interpolate_p

            p_lo, self.left = interpolate_p(
                p=np.linspace(0.0001, 0.9999, len(self.left)), q=self.left
            )
            p_hi, self.right = interpolate_p(
                p=np.linspace(0.0001, 0.9999, len(self.right)), q=self.right
            )

    # * --------------------- operators ---------------------*#

    def __iter__(self):
        return iter(self.to_interval())

    # * --------------------- functions ---------------------*#
    def to_interval(self):
        from .intervals.number import Interval as I

        return I(lo=self.left, hi=self.right)


class Staircase(Box):
    """distribution free p-box"""

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
        self.var = I(self.var_lo, self.var_hi)

    def __repr__(self):
        mean_text = f"{self.mean}"
        var_text = f"{self.var}"
        range_text = f"{self._range}"
        return f"Pbox ~ (range={range_text}, mean={mean_text}, var={var_text})"

    def plot(
        self,
        title=None,
        ax=None,
        style="band",
        fill_color="lightgray",
        bound_colors=None,
        **kwargs,
    ):
        """default plotting function"""

        if ax is None:
            fig, ax = plt.subplots()

        p_axis = self._pvalues if self._pvalues is not None else Params.p_values
        plot_bound_colors = bound_colors if bound_colors is not None else ["g", "b"]

        def display_the_box():
            """display two F curves plus the top-bottom horizontal lines"""
            ax.step(self.left, p_axis, c=plot_bound_colors[0], where="post")
            ax.step(self.right, p_axis, c=plot_bound_colors[1], where="post")
            ax.plot([self.left[0], self.right[0]], [0, 0], c=plot_bound_colors[0])
            ax.plot([self.left[-1], self.right[-1]], [1, 1], c=plot_bound_colors[1])

        if title is not None:
            ax.set_title(title)
        if style == "band":
            ax.fill_betweenx(
                y=p_axis,
                x1=self.left,
                x2=self.right,
                interpolate=True,
                color=fill_color,
                alpha=0.3,
                **kwargs,
            )
            display_the_box()
        elif style == "simple":
            display_the_box()
        else:
            raise ValueError("style must be either 'simple' or 'band'")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\Pr(X \leq x)$")
        "label" in kwargs and ax.legend()
        return ax

    def display(self):
        self.plot()
        plt.show()

    # * --------------------- constructors ---------------------*#
    @classmethod
    def from_CDFbundle(cls, a, b):
        """pbox from emipirical CDF bundle
        args:
            - a : CDF bundle of lower extreme F;
            - b : CDF bundle of upper extreme F;
        """
        from .constructors import interpolate_p

        p_lo, q_lo = interpolate_p(a.probabilities, a.quantiles)
        p_hi, q_hi = interpolate_p(b.probabilities, b.quantiles)
        return cls(left=q_lo, right=q_hi, p_values=p_lo)

    # * --------------------- operators ---------------------*#
    def __add__(self):
        pass

    # * --------------------- methods ---------------------*#
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

    # as comparison
    def add_old(self, other, method="f"):
        """addtion of uncertain numbers with the defined dependency method"""

        from .pbox_base import Pbox

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
        print(len(nleft))
        nleft.sort()
        nright.sort()

        return Pbox(left=nleft, right=nright, steps=self.steps)


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
        return f"{base_repr}, shape={self.shape}{self.dist_params}"

    def sample():
        pass


class Cbox(Box):
    def __init__(self, left, right, steps=200):
        super().__init__(left, right, steps)


# * --------------------- module functions ---------------------*#


def is_un(un):
    """if the `un` is modelled by accepted constructs"""

    from .intervals.number import Interval
    from .ds import DempsterShafer
    from .distributions import Distribution

    return isinstance(un, Box | Interval | DempsterShafer | Distribution)


def convert_pbox(un):
    """transform the input un into a Pbox object

    note:
        - theorically 'un' can be {Interval, DempsterShafer, Distribution, float, int}
    """

    from .pbox_base import Pbox
    from .interval import Interval as nInterval
    from .ds import DempsterShafer
    from .distributions import Distribution
    from .intervals.number import Interval as I

    if isinstance(un, Box):
        return un
    elif isinstance(un, nInterval):
        return Pbox(un.left, un.right)
    elif isinstance(un, I):
        return Staircase(
            left=np.repeat(un.lo, Params.steps),
            right=np.repeat(un.hi, Params.steps),
            mean=un,
            var=I(0, (un.hi - un.lo) * (un.hi - un.lo) / 4),
        )
    elif isinstance(un, Pbox):
        return un
    elif isinstance(un, Distribution):
        return un.to_pbox()
    elif isinstance(un, DempsterShafer):
        return un.to_pbox()
    else:
        raise TypeError(f"Unable to convert {type(un)} object to Pbox")


def pbox_number_ops(pbox: Staircase | Leaf, n: float | int, f: callable):
    """blueprint for arithmetic between pbox and real numbers"""
    l = f(pbox.left, n)
    r = f(pbox.right, n)
    new_mean = f(pbox.mean, n)
    return Staircase(left=l, right=r, mean=new_mean, var=pbox.var)

    # Staircase(left=pbox.left + n, right=pbox.right + n)
