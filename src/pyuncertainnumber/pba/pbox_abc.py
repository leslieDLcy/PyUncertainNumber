import numpy as np
from abc import ABC, abstractmethod
from .pbox_base import _sideVariance
from .utils import is_increasing
from .params import Params
import matplotlib.pyplot as plt
from .intervals.number import Interval as I


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
        self, left: np.ndarray | list, right: np.ndarray | list, steps, p_values=None
    ):
        self.left = np.array(left)
        self.right = np.array(right)
        self.steps = steps
        self._pvalues = p_values if p_values is not None else Params.p_values
        self._init_moments_range()

    @abstractmethod
    def _init_moments_range(self):
        pass

    def post_init_check(self):

        assert len(self.left) == len(
            self.right
        ), "steps of lower/upper bounds not consistent"

        if (not is_increasing(self.left)) or (not is_increasing(self.right)):
            raise Exception("Left and right arrays must be increasing")


class Staircase(Box):
    """distribution free p-box"""

    def __init__(
        self,
        left,
        right,
        steps=200,
        p_values=None,
    ):
        super().__init__(left, right, steps, p_values)

    @abstractmethod
    def _init_moments_range(self):
        """initialised `mean`, `var` and `range` bounds"""

        #! should we compute mean if it is a Cauchy, var if it's a t distribution?
        #! we assume that two extreme bounds are valid CDFs
        self.mean_lo, self.var_lo = get_var_from_ecdf(self.left, self._pvalues)
        self.mean_hi, self.var_hi = get_var_from_ecdf(self.right, self._pvalues)
        self.mean = I(self.mean_lo, self.mean_hi)
        self.var = I(self.var_lo, self.var_hi)
        self._range = I(min(self.left), max(self.right))

    def __repr__(self):
        # with np.printoptions(precision=2, suppress=True):
        # mean_text = f"[{self.mean_lo:.2f}, {self.mean_hi:.2f}]"
        # var_text = f"[{self.var_lo:.2f}, {self.var_hi:.2f}]"
        mean_text = f"{self.mean}"
        var_text = f"{self.var}"
        range_text = f"{self._range}"
        return f"Pbox ~ (range={range_text}, mean={mean_text}, var={var_text})"

    def display(
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
            ax.plot([self.left[0], self.right[0]], [0, 0], c="b")
            ax.plot([self.left[-1], self.right[-1]], [1, 1], c="g")

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


class Leaf(Staircase):
    """parametric pbox"""

    def __init__(
        self,
        shape=None,
        left=None,
        right=None,
        mean=None,
        var=None,
        dist_params=None,
        steps=200,
    ):
        super().__init__(left, right, steps)
        self.shape = shape
        self.dist_params = dist_params
        self.mean = mean
        self.var = var

    def _init_moments_range(self):
        self._range = I(min(self.left).item(), max(self.right).item())

    def __repr__(self):
        base_repr = super().__repr__().rstrip(")")  # remove trailing ')'
        return f"{base_repr}, shape={self.shape}{self.dist_params}"

    def sample():
        pass


class Cbox(Box):
    def __init__(self, left, right, steps=200):
        super().__init__(left, right, steps)
