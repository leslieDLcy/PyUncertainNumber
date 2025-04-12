import numpy as np
from abc import ABC, abstractmethod
from .pbox_base import _sideVariance
from .utils import is_increasing
from .params import Params
import matplotlib.pyplot as plt


def round_numbers(l: list):
    if isinstance(l, list):
        return [round(n, 3) for n in l]
    elif isinstance(l, float | int):
        return round(l, 3)
    else:
        raise Exception("Not implemented")


class Box(ABC):
    """a base class for Pbox"""

    def __init__(self, left: np.ndarray | list, right: np.ndarray | list, steps):
        self.left = np.array(left)
        self.right = np.array(right)
        self.steps = steps
        self._init_moments_range()

    def _init_moments_range(self):
        """initialised `mean`, `var` and `range` bounds"""
        # TODO revise variance computation later on
        # should we compute mean if it is a Cauchy, var if it's a t distribution?
        self.mean_left = np.mean(self.left)
        self.mean_right = np.mean(self.right)

        self.var_left = 0.0

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

        self._range = [min(self.left).item(), max(self.right).item()]

    def post_init_check(self):

        assert len(self.left) == len(
            self.right
        ), "steps of lower/upper bounds not consistent"

        if (not is_increasing(self.left)) or (not is_increasing(self.right)):
            raise Exception("Left and right arrays must be increasing")


class Staircase(Box):
    """distribution free p-box"""

    def __init__(self, left, right, p_values=None, steps=200):
        super().__init__(left, right, steps)
        self._pvalues = p_values

    def __repr__(self):
        mean_text = (
            f"[{round_numbers(self.mean_left)}, {round_numbers(self.mean_right)}]"
        )
        var_text = f"[{round_numbers(self.var_left)}, {round_numbers(self.var_right)}]"
        range_text = f"{round_numbers(self._range)}"
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
        return cls(left=q_lo, right=q_hi)


class Leaf(Staircase):
    """parametric pbox"""

    def __init__(self, shape, left, right, dist_params, steps=200):
        super().__init__(left, right, steps)
        self.shape = shape
        self.dist_params = dist_params

    def __repr__(self):
        base_repr = super().__repr__().rstrip(")")  # remove trailing ')'
        return f"{base_repr}, shape={self.shape})"

    def sample():
        pass


class Cbox(Box):
    def __init__(self, left, right, steps=200):
        super().__init__(left, right, steps)
