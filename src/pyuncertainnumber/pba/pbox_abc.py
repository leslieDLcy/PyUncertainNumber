from __future__ import annotations
from typing import TYPE_CHECKING
import warnings
import numpy as np
from abc import ABC, abstractmethod
from .params import Params
import matplotlib.pyplot as plt
from .intervals.number import Interval as I
from numbers import Number
import operator
from .utils import (
    condensation,
    find_nearest,
    get_mean_var_from_ecdf,
    is_increasing,
    left_right_switch,
    variance_bounds_via_lp,
    area_between_ecdfs,
)
import logging
from .context import get_current_dependency
from .mixins import NominalValueMixin
from contextlib import suppress

if TYPE_CHECKING:
    from pyuncertainnumber import Interval
    from .dss import DempsterShafer
    from .ecdf import eCDF_bundle
    from typing import Self

# Configure the logging system with a simple format
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


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


def pbox_from_ecdf_bundle(lower_bound: eCDF_bundle, upper_bound: eCDF_bundle) -> Pbox:
    """Construct a p-box from two empirical CDF bundles as the extreme bounds"""
    return Staircase.from_CDFbundle(lower_bound, upper_bound)


def naive_frechet_pbox(x, y, op) -> Staircase:
    """A wrapper that returns a Pbox from the naive Frechet operation

    note:
        old implementation from pba.r
    """
    from .operation import naive_frechet_op

    Zu, Zd = naive_frechet_op(x, y, op)
    p = Staircase(left=Zu, right=Zd)
    return p


# deprecated
# def new_naive_frechet_pbox(x, y, op) -> Staircase:
#     """A wrapper that returns a Pbox from the naive Frechet operation"""
#     from .operation import new_naive_frechet_op

#     Zu, Zd = new_naive_frechet_op(x, y, op)
#     p = Staircase(left=Zu, right=Zd)
#     return p


def vectorised_naive_frechet_pbox(x, y, op) -> Staircase:
    """A wrapper that returns a Pbox from the naive Frechet operation"""
    from .operation import new_vectorised_naive_frechet_op

    Zu, Zd = new_vectorised_naive_frechet_op(x, y, op)
    p = Staircase(left=Zu, right=Zd)
    return p


def classic_frechet_pbox(x, y, op) -> Staircase:
    """this corresponds to the Frank, Nelson and Sklar Frechet bounds implementation"""
    from .operation import frechet_op

    left, right = frechet_op(x, y, op)
    p = Staircase(left=left, right=right)
    return p


def straddle_frechet_pbox(x, y):
    """bespoke Frechet for multiplcation when anyone straddles 0"""
    from .aggregation import imposition

    warnings.warn(
        "Multiplication of a pbox straddling zero needs attention",
        UserWarning,
    )
    naive_base_p = vectorised_naive_frechet_pbox(x, y, operator.mul)
    balch_p = x.balchprod(y)
    imp_p = imposition(naive_base_p, balch_p)
    return imp_p


def nagative_frechet_pbox(x, y):
    if x.hi <= 0 or y.hi <= 0:
        warnings.warn("negative pbox encountered")
        a = -x if x.hi <= 0 else x
        b = -y if y.hi <= 0 else y
        result = classic_frechet_pbox(a, b, operator.mul)
        return -result if (x.hi <= 0) ^ (y.hi <= 0) else result
    else:
        raise Exception("Not nagative Frechet case")


def frechet_pbox_mul(x, y):
    """the overall pbox"""

    if x.straddles_zero() or y.straddles_zero():  # if any one straddles
        if y.straddles_zero():  # y shall be straddle
            return straddle_frechet_pbox(x, y)
        else:
            return straddle_frechet_pbox(y, x)
    # elif x.is_zero() or y.is_zero():
    #     return 0
    elif x.hi <= 0 or y.hi <= 0:
        return nagative_frechet_pbox(x, y)
    else:  # both positive
        return classic_frechet_pbox(x, y, operator.mul)


class Pbox(NominalValueMixin, ABC):
    """a base class for Pbox

    danger:
        this is an abstract class and should not be instantiated directly.

        .. seealso::

            :class:`pbox_abc.Staircase` and :class:`pbox_abc.Leaf` for concrete implementations.
    """

    def __init__(
        self,
        left: np.ndarray | list,
        right: np.ndarray | list,
        steps=Params.steps,
        mean=None,
        var=None,
        p_values=None,
    ):
        left, right = left_right_switch(left, right)
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
        if (self.mean is None) or (self.var is None):
            self._init_moments()

        self._init_range()

        self.degenerate_flag()

    def steps_check(self):

        assert len(self.left) == len(
            self.right
        ), "Length of lower and upper bounds is not consistent"

    def _compute_nominal_value(self):
        return np.round(self.mean.mid, 3)

    def degenerate_flag(self) -> bool:
        """check if the pbox is degenerate (i.e. left == right everywhere)"""
        self._degenerate = np.array_equal(self.left, self.right)

    @property
    def degenerate(self) -> bool:
        return self._degenerate

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
    def enclosed_area(self):
        """the enclosed area between the two extreme cdfs"""
        return area_between_ecdfs(
            x_upper=self.left,
            p_upper=self._pvalues,
            x_lower=self.right,
            p_lower=self._pvalues,
        )

    # * --------------------- operators ---------------------*#

    def __iter__(self):
        return iter(self.to_interval())

    def __eq__(self, other):
        """Equality operator for Pbox objects

        note:
            - two pboxes are equal if their left and right bounds are equal
        """
        # equal = np.array_equal(self.left, other.left) and np.array_equal(
        #     self.right, other.right
        # )
        close = np.allclose(self.left, other.left) and np.allclose(
            self.right, other.right
        )
        return close

    def __contains__(self, item):
        if isinstance(item, Number):
            return (self.lo <= item) and (item <= self.hi)
        else:
            return (self.lo <= item.lo) and (item.hi <= self.hi)

    # * --------------------- functions ---------------------*#
    def to_interval(self):
        """discretise pbox into a vec-interval of length of default steps

        note:
            If desired a custom length of vec-interval as output, use `discretise()` method.
        """
        from .intervals.number import Interval as I

        return I(lo=self.left, hi=self.right)

    def to_dss(self, discretisation=Params.steps):
        """convert pbox to DempsterShafer object"""
        from .dss import DempsterShafer

        return DempsterShafer(
            self.to_interval(),
            np.repeat(a=(1 / discretisation), repeats=discretisation),
        )

    def to_numpy(self):
        """convert pbox to a 2D numpy array (n, 2) of left and right"""
        return np.stack((self.left, self.right), axis=1)


class Staircase(Pbox):
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
        """Initialize mean/var interval estimates.

        strategy:
            1) Try LP-based bounds.
            2) If that fails, try ECDF-based bounds.
            3) If that also fails, set to NaN intervals so the program continues.

        This function NEVER raises.
        """
        # Defaults
        mean_I = None
        var_I = None
        method_used = None
        errors = []

        # --- Attempt 1: LP bounds ---
        try:
            dict_moments = variance_bounds_via_lp(
                q_a=self.left,
                p_a=self._pvalues,
                q_b=self.right,
                p_b=self._pvalues,
                x_grid=np.linspace(self.lo, self.hi, 50),
            )
            self.mean_lo, self.mean_hi = dict_moments["mu_min"], dict_moments["mu_max"]
            self.var_lo, self.var_hi = dict_moments["var_min"], dict_moments["var_max"]
            mean_I = I(self.mean_lo, self.mean_hi)
            var_I = I(self.var_lo, self.var_hi)
            method_used = "lp_bounds"
        except Exception as e:
            errors.append(("lp_bounds", repr(e)))

        # --- Attempt 2: ECDF fallback (only if needed) ---
        if mean_I is None or var_I is None:
            try:
                mean_lo, var_lo = get_mean_var_from_ecdf(self.left, self._pvalues)
                mean_hi, var_hi = get_mean_var_from_ecdf(self.right, self._pvalues)
                self.mean_lo, self.mean_hi = mean_lo, mean_hi
                self.var_lo, self.var_hi = var_lo, var_hi
                mean_I = I(self.mean_lo, self.mean_hi)
                var_I = I(self.var_lo, self.var_hi)
                method_used = "ecdf_fallback"
            except Exception as e:
                errors.append(("ecdf_fallback", repr(e)))

        # --- Last resort: make it unambiguous and safe ---
        if mean_I is None or var_I is None:
            # Use NaN to signal “unknown/unavailable” without risking real-number collisions.
            mean_I = I(666, 666)
            var_I = I(666, 666)
            method_used = method_used or "unavailable"

        # --- Assign + annotate; nothing in here may raise ---
        self.mean = mean_I
        self.var = var_I
        # Optional: stash debug info for inspection; never let this crash.
        with suppress(Exception):
            self._moments_meta = {
                "method": method_used,
                "errors": errors,  # list of (stage, repr(error))
                "lo_hi": {
                    "mean": (
                        getattr(self, "mean_lo", None),
                        getattr(self, "mean_hi", None),
                    ),
                    "var": (
                        getattr(self, "var_lo", None),
                        getattr(self, "var_hi", None),
                    ),
                },
            }

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
        bound_styles=None,  # e.g. ("--", ":")
        left_line_kwargs=None,  # e.g. {"linewidth": 2, "alpha": 0.9}
        right_line_kwargs=None,  # e.g. {"linewidth": 2, "alpha": 0.9}
        nuance="step",
        alpha=0.3,
        **kwargs,
    ):
        """default plotting function

        args:
            style (str): 'box' or 'simple'
            fill_color (str): color to fill the box (only for 'box' style)
            bound_colors (list): list of two colors for left and right bound lines
            bound_styles (list): list of two linestyles for left and right bound lines
            left_line_kwargs (dict): additional kwargs for left bound line
            right_line_kwargs (dict): additional kwargs for right bound line
            nuance (str): 'step' or 'curve' for bound line styles
            alpha (float): transparency level for the box fill (only for 'box' style)
            **kwargs: additional keyword arguments for the plot


        note:
            Two styles are supported: a 'box' with fill-in color and a 'simple' one without fill-in color.
            Color and linestyle of the bound lines can be customized via the `bound_styles`, `left_line_kwargs`, and `right_line_kwargs` parameters.
            The argument `nuance` controls whether the bound lines are plotted as step functions ('step') or smooth curves ('curve').


        example:
            >>> a = pba.normal([2, 6], [0.5, 1])
            >>> fig, ax = plt.subplots()
            >>> a.plot(ax=ax, style='simple')  # simple style without fill-in color
            >>> # box style with fill-in color and also customized bound colors
            >>> a.plot(ax=ax, style='box',
            ...     fill_color='lightblue',
            ...     bound_colors = ['lightblue', 'lightblue'],
            ...     bound_styles=("--", ":"),
            ...     alpha=0.5
            ... )
            >>> # customized left and right bound line styles
            >>> ax = pbox.plot(
            ...     left_line_kwargs={"linestyle": "--", "linewidth": 2},
            ...     right_line_kwargs={"linestyle": ":", "linewidth": 2, "alpha": 0.8},
            )

        """
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe  # optional; for "shaded/halo" line effects
        from .utils import CustomEdgeRectHandler

        if ax is None:
            fig, ax = plt.subplots()

        p_axis = self._pvalues if self._pvalues is not None else Params.p_values
        plot_bound_colors = bound_colors if bound_colors is not None else ["g", "b"]

        # defaults
        if bound_styles is None:
            bound_styles = ("solid", "solid")
        left_line_kwargs = {} if left_line_kwargs is None else dict(left_line_kwargs)
        right_line_kwargs = {} if right_line_kwargs is None else dict(right_line_kwargs)

        # ensure color + linestyle are set unless user overrode them
        left_defaults = {"c": plot_bound_colors[0], "linestyle": bound_styles[0]}
        right_defaults = {"c": plot_bound_colors[1], "linestyle": bound_styles[1]}
        # user kwargs take precedence
        left_kwargs = {**left_defaults, **left_line_kwargs}
        right_kwargs = {**right_defaults, **right_line_kwargs}

        def display_box(nuance, label=None):
            """display two F curves plus the top-bottom horizontal lines"""

            if nuance == "step":
                step_kwargs_left = {"where": "post", **left_kwargs}
                step_kwargs_right = {"where": "post", **right_kwargs}
                if label is not None:
                    step_kwargs_left["label"] = label

                (line_left,) = ax.step(self.left, p_axis, **step_kwargs_left)
                (line_right,) = ax.step(self.right, p_axis, **step_kwargs_right)
            elif nuance == "curve":
                curve_kwargs_left = {**left_kwargs}
                curve_kwargs_right = {**right_kwargs}
                if label is not None:
                    curve_kwargs_left["label"] = label

                (line_left,) = ax.plot(self.left, p_axis, **curve_kwargs_left)
                (line_right,) = ax.plot(self.right, p_axis, **curve_kwargs_right)
            else:
                raise ValueError("nuance must be either 'step' or 'curve'")

            # horizontal caps (use right/left kwargs for consistent style/color)
            ax.plot([self.left[0], self.right[0]], [0, 0], **right_kwargs)
            ax.plot([self.left[-1], self.right[-1]], [1, 1], **left_kwargs)

            if label is not None:
                ax.legend(
                    handler_map={line_left: CustomEdgeRectHandler()}
                )  # regular use

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
            display_box(nuance, label=kwargs.get("label"))
        else:
            raise ValueError("style must be either 'simple' or 'box'")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\Pr(X \leq x)$")
        return ax

    # backup old plot function
    # def plot(
    #     self,
    #     title=None,
    #     ax=None,
    #     style="box",
    #     fill_color="lightgray",
    #     bound_colors=None,
    #     nuance="step",
    #     alpha=0.3,
    #     **kwargs,
    # ):
    #     """default plotting function

    #     args:
    #         style (str): 'box' or 'simple'
    #     """
    #     from .utils import CustomEdgeRectHandler

    #     if ax is None:
    #         fig, ax = plt.subplots()

    #     p_axis = self._pvalues if self._pvalues is not None else Params.p_values
    #     plot_bound_colors = bound_colors if bound_colors is not None else ["g", "b"]

    #     def display_box(nuance, label=None):
    #         """display two F curves plus the top-bottom horizontal lines"""

    #         if nuance == "step":
    #             step_kwargs = {
    #                 "c": plot_bound_colors[0],
    #                 "where": "post",
    #             }

    #             if label is not None:
    #                 step_kwargs["label"] = label

    #             # Make the plot
    #             (line,) = ax.step(self.left, p_axis, **step_kwargs)
    #             ax.step(self.right, p_axis, c=plot_bound_colors[1], where="post")
    #             ax.plot([self.left[0], self.right[0]], [0, 0], c=plot_bound_colors[1])
    #             ax.plot([self.left[-1], self.right[-1]], [1, 1], c=plot_bound_colors[0])
    #         elif nuance == "curve":
    #             smooth_curve_kwargs = {
    #                 "c": plot_bound_colors[0],
    #             }

    #             if label is not None:
    #                 smooth_curve_kwargs["label"] = label

    #             (line,) = ax.plot(self.left, p_axis, **smooth_curve_kwargs)
    #             ax.plot(self.right, p_axis, c=plot_bound_colors[1])
    #             ax.plot([self.left[0], self.right[0]], [0, 0], c=plot_bound_colors[1])
    #             ax.plot([self.left[-1], self.right[-1]], [1, 1], c=plot_bound_colors[0])
    #         else:
    #             raise ValueError("nuance must be either 'step' or 'curve'")
    #         if label is not None:
    #             ax.legend(handler_map={line: CustomEdgeRectHandler()})  # regular use

    #     if title is not None:
    #         ax.set_title(title)
    #     if style == "box":
    #         ax.fill_betweenx(
    #             y=p_axis,
    #             x1=self.left,
    #             x2=self.right,
    #             interpolate=True,
    #             color=fill_color,
    #             alpha=alpha,
    #             **kwargs,
    #         )
    #         display_box(nuance, label=None)
    #         if "label" in kwargs:
    #             ax.legend(loc="best")
    #     elif style == "simple":
    #         display_box(nuance, label=kwargs["label"] if "label" in kwargs else None)
    #     else:
    #         raise ValueError("style must be either 'simple' or 'box'")
    #     ax.set_xlabel(r"$x$")
    #     ax.set_ylabel(r"$\Pr(X \leq x)$")
    #     return ax

    #### put something below ####
    def plot_reverse_axis(
        self,
        title=None,
        ax=None,
        style="box",
        fill_color="lightgray",
        bound_colors=None,
        nuance="step",
        alpha=0.3,
        orientation="xy",  # NEW: "xy" (default) or "yx" (swap axes)
        invert_xaxis=True,
        **kwargs,
    ):
        """A testing plotting function that can swap quantile and probability axes.

        args:
            style (str): 'box' or 'simple'
            orientation (str): 'xy' keeps x on horizontal and Pr(X<=x) on vertical;
                            'yx' swaps them.
        """
        from .utils import CustomEdgeRectHandler
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        p_axis = self._pvalues if self._pvalues is not None else Params.p_values
        plot_bound_colors = bound_colors if bound_colors is not None else ["g", "b"]

        def display_box(nuance, label=None):
            """display two F curves plus the top-bottom horizontal/vertical lines,
            depending on orientation.
            """
            if orientation == "xy":
                # x = left/right, y = p
                if nuance == "step":
                    step_kwargs = {"c": plot_bound_colors[0], "where": "post"}
                    if label is not None:
                        step_kwargs["label"] = label
                    (line,) = ax.step(self.left, p_axis, **step_kwargs)
                    ax.step(self.right, p_axis, c=plot_bound_colors[1], where="post")
                elif nuance == "curve":
                    curve_kwargs = {"c": plot_bound_colors[0]}
                    if label is not None:
                        curve_kwargs["label"] = label
                    (line,) = ax.plot(self.left, p_axis, **curve_kwargs)
                    ax.plot(self.right, p_axis, c=plot_bound_colors[1])
                else:
                    raise ValueError("nuance must be either 'step' or 'curve'")
                ax.plot([self.left[0], self.right[0]], [0, 0], c=plot_bound_colors[1])
                ax.plot([self.left[-1], self.right[-1]], [1, 1], c=plot_bound_colors[0])

            elif orientation == "yx":
                # x = p, y = left/right  (axes swapped)
                if nuance == "step":
                    step_kwargs = {"c": plot_bound_colors[0], "where": "post"}
                    if label is not None:
                        step_kwargs["label"] = label
                    (line,) = ax.step(p_axis, self.left, **step_kwargs)
                    ax.step(p_axis, self.right, c=plot_bound_colors[1], where="post")
                elif nuance == "curve":
                    curve_kwargs = {"c": plot_bound_colors[0]}
                    if label is not None:
                        curve_kwargs["label"] = label
                    (line,) = ax.plot(p_axis, self.left, **curve_kwargs)
                    ax.plot(p_axis, self.right, c=plot_bound_colors[1])
                else:
                    raise ValueError("nuance must be either 'step' or 'curve'")
                ax.plot([0, 0], [self.left[0], self.right[0]], c=plot_bound_colors[1])
                ax.plot([1, 1], [self.left[-1], self.right[-1]], c=plot_bound_colors[0])
            else:
                raise ValueError("orientation must be 'xy' or 'yx'")

            if label is not None:
                ax.legend(handler_map={line: CustomEdgeRectHandler()})

        if title is not None:
            ax.set_title(title)

        if style == "box":
            if orientation == "xy":
                ax.fill_betweenx(
                    y=p_axis,
                    x1=self.left,
                    x2=self.right,
                    interpolate=True,
                    color=fill_color,
                    alpha=alpha,
                    **kwargs,
                )
            else:  # 'yx'
                ax.fill_between(
                    x=p_axis,
                    y1=self.left,
                    y2=self.right,
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

        if orientation == "xy":
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$\Pr(X \leq x)$")
        else:
            ax.set_xlabel(r"$\Pr(X \leq x)$")
            ax.set_ylabel(r"$x$")
            if invert_xaxis:
                ax.invert_xaxis()  # NEW LINE — reverses new x-axis (1 → 0)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

        return ax

    ### above ###

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

    def plot_quantile_bound(self, p: float, ax=None, **kwargs):
        """plot the quantile bound at a certain probability level p

        note:
            - a horizontal line
        """
        if ax is None:
            fig, ax = plt.subplots()

        x_lo, x_hi = self.alpha_cut(p).lo, self.alpha_cut(p).hi

        self.plot(ax=ax, **kwargs)

        ax.plot(
            [x_lo, x_hi],
            [p, p],
            c="r",
            label="probability bound",
            zorder=50,
        )
        ax.scatter(x_lo, p, c="r", marker=">", zorder=50)
        ax.scatter(x_hi, p, c="r", marker="<", zorder=50)
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
        return self.add(other, dependency=get_current_dependency())

    def __radd__(self, other):
        return self.add(other, dependency=get_current_dependency())

    def __sub__(self, other):
        return self.sub(other, dependency=get_current_dependency())

    def __rsub__(self, other):
        return (-self).add(other, dependency=get_current_dependency())

    def __mul__(self, other):
        return self.mul(other, dependency=get_current_dependency())

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):

        return self.div(other, dependency=get_current_dependency())

    def __rtruediv__(self, other):
        try:
            return other * self.reciprocal()
        except:
            return NotImplemented

    def __pow__(self, other):
        return self.pow(other, dependency=get_current_dependency())

    def __rpow__(self, other: Number):
        """Power operation with the base as `other` and self as the exponent"""
        from functools import partial

        bar = partial(np.power, other)  # other as the base
        return self._unary_template(bar)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented
        if len(inputs) != 1 or inputs[0] is not self:
            return NotImplemented
        if "out" in kwargs and kwargs["out"] is not None:
            return NotImplemented

        if ufunc is np.sin:
            return self.sin()
        if ufunc is np.cos:
            return self.cos()
        if ufunc is np.tanh:
            return self.tanh()
        if ufunc is np.exp:
            return self.exp()
        if ufunc is np.sqrt:
            return self.sqrt()
        if ufunc is np.log:
            return self.log()
        if ufunc is np.reciprocal:
            return self.reciprocal()

        return NotImplemented

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
        """test the lightweight `alpha_cut` method

        args:
            alpha (array-like): probability levels
        """
        from .intervals.number import LightweightInterval as lwI
        from .intervals.number import Interval as I

        ind = find_nearest(Params.p_values, alpha)
        return I(lo=self.left[ind], hi=self.right[ind])

    def sample(self, n_sam):
        """LHS sampling by default"""
        from scipy.stats import qmc

        alpha = np.squeeze(qmc.LatinHypercube(d=1).random(n=n_sam))
        return self.alpha_cut(alpha)

    def discretise(self, n=None) -> Interval:
        """alpha-cut discretisation of the p-box without outward rounding

        args:
            n (int): number of steps to be used in the discretisation.

        return:
            vector Interval
        """

        if (n is None) or (n == Params.steps):
            return I(lo=self.left, hi=self.right)
        else:
            p_values = np.linspace(Params.p_lboundary, Params.p_hboundary, n)
            return self.alpha_cut(p_values)

    def outer_discretisation(self, n=None):
        """discretisation of a p-box to get intervals based on the scheme of outer approximation

        args:
            n (int): number of steps to be used in the discretisation

        note:
            `the_interval_list` will have length one less than that of default `p_values` (i.e. 100 and 99)

        return:
            the outer intervals in vec-Interval form
        """

        from .intervals.number import Interval as I
        from .intervals.number import LightweightInterval as lwI

        if n is not None:
            p_values = np.linspace(Params.p_lboundary, Params.p_hboundary, n)
        else:
            p_values = self._pvalues

        p_leftend = p_values[0:-1]
        p_rightend = p_values[1:]

        q_l = self.alpha_cut(p_leftend).left
        q_r = self.alpha_cut(p_rightend).right
        interval_vec = lwI(lo=q_l, hi=q_r)

        return interval_vec

    def condensation(self, n) -> Self:
        """ourter condensation of the pbox to reduce the number of steps and get a sparser staircase pbox

        args:
            n (int): number of steps to be used in the discretisation

        note:
            Have not thought about a better name so we call it `condensation` for now. Candidate names include 'approximation'.
            It will ouput a p-box and keep steps as 200 for computational consistency.

        example:
            >>> p.condensation(n=5)

        return:
            a staircase p-box that looks sparser but has the same number of steps
        """
        from .aggregation import stacking

        itvls = self.outer_discretisation(n)
        return stacking(itvls)

    def condense(self, n) -> DempsterShafer:
        """Another condensation function which has steps of n

        Compared to the above `condensation` method that ouputs a p-box and  keeps steps as 200 for computational consistency.
        This one condenses in a more literal manner, as in having n steps in the resulting Dempster-Shafer structure.
        """
        from .intervals.number import Interval as I
        from .dss import DempsterShafer as DSS

        condensed_x = self.condensation(n)

        # condensed bounds
        con_bd_l = np.unique(condensed_x.left)
        con_bd_r = np.unique(condensed_x.right)

        real_condensed_dss_x = DSS(
            intervals=I(
                lo=con_bd_l,
                hi=con_bd_r,
            ),
            masses=[1 / len(con_bd_l)] * len(con_bd_l),
        )
        return real_condensed_dss_x

    def truncate(self, a, b):
        """Truncate the Pbox to the range [a, b].

        example:
        >>> from pyuncertainnumber import pba
        >>> p = pba.normal([4, 9], 1)
        >>> tr = p.truncate(3, 8)
        >>> fig, ax = plt.subplots()
        >>> p.plot(ax=ax)
        >>> tr.plot(ax=ax, fill_color='r')
        >>> plt.show()
        """

        from .aggregation import _imposition
        from .pbox_free import min_max

        i = min_max(a, b, return_construct=True)
        return _imposition(self, i)

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

    def get_PI(self, alpha: Number = 0.95, style="narrowest") -> Interval:
        """Compute the predictive interval at the coverage level of `alpha`

        args:
            alpha (Number): coverage level for the predictive interval, default is 0.95
            style (str): 'narrowest' or 'widest', default is 'narrowest'

        note:
            by default, narrowest predictive interval is returned;
            when the narrowest does not exist, a warning will the generated and then the widest is returned instead.

        example:
            >>> from pyuncertainnumber import pba
            >>> p = pba.normal([10, 15, 1])
            >>> p.get_PI(alpha=0.95, style='narrowest')
        """
        from pyuncertainnumber import Interval

        lo_cut_level = (1 - alpha) / 2  # 0.025
        hi_cut_level = 1 - lo_cut_level  # 0.975

        if style == "narrowest":
            hi = self.alpha_cut(hi_cut_level).lo
            lo = self.alpha_cut(lo_cut_level).hi
            try:
                return Interval(lo=lo, hi=hi)
            except Exception:
                logging.warning(
                    "narrowest predictive interval does not exist. Return 'widest' style instead."
                )
                # do the style == 'widest'
                hi = self.alpha_cut(hi_cut_level).hi
                lo = self.alpha_cut(lo_cut_level).lo
                return Interval(lo=lo, hi=hi)
        elif style == "widest":
            hi = self.alpha_cut(hi_cut_level).hi
            lo = self.alpha_cut(lo_cut_level).lo
            return Interval(lo=lo, hi=hi)

    # * --------------------- states --------------------- *#

    def straddles(self, N, endpoints=True) -> bool:
        """Check whether the p-box straddles a number N

        args:
            N (float): the Number to check
            endpoints (Boolean): Whether to include the endpoints within the check

        return:
            True
                If :math:`\\mathrm{left} \\leq N \\leq \mathrm{right}` (Assuming `endpoints=True`)
            False
                Otherwise

        note:
            This could affect the results of Frechet bounds
        """
        if endpoints:
            if min(self.left) <= N and max(self.right) >= N:
                return True
        else:
            if min(self.left) < N and max(self.right) > N:
                return True

        return False

    def straddles_zero(self) -> bool:
        """Checks specifically whether :math:`0` is within the p-box"""
        return self.straddles(0, False)

    def is_zero(self):
        return self.lo == 0

    def is_nagative(self):
        return self.hi <= 0

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
                raise Exception(
                    "Imposition does not exist as high left greater than low right"
                )
            u.append(max(sL, oL))
            d.append(min(sR, oR))
        return Staircase(left=u, right=d)

    # * ---------------------unary operations--------------------- *#

    def _unary_template(self, f):
        l, r = f(self.left), f(self.right)
        return Staircase(left=l, right=r)

    def exp(self):
        """exponential function: e^x"""
        return self._unary_template(np.exp)

    def sqrt(self):
        """square root function: √x"""
        return self._unary_template(np.sqrt)

    def reciprocal(self):
        """Calculate the reciprocal of the pbox

        note:
            the pbox should not straddle zero, otherwise a warning is raised
        """

        if self.straddles_zero():
            warnings.warn(
                "Division of a pbox straddling zero needs attention", UserWarning
            )
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
        from .operation import (
            frechet_op,
            independent_op,
            perfect_op,
            opposite_op,
        )

        if isinstance(other, Number):
            return pbox_number_ops(self, other, operator.add)
        if is_un(other):
            other = convert_pbox(other)
        match dependency:
            case "f":
                nleft, nright = frechet_op(self, other, operator.add)
            case "p":
                nleft, nright = perfect_op(self, other, operator.add)
            case "o":
                nleft, nright = opposite_op(self, other, operator.add)
            case "i":
                nleft, nright = independent_op(self, other, operator.add)
            case _:
                raise ValueError(f"Unknown dependency type: {dependency!r}")
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
        from .operation import (
            independent_op,
            perfect_op,
            opposite_op,
        )

        if isinstance(other, Number):
            return pbox_number_ops(self, other, operator.mul)
        if is_un(other):
            other = convert_pbox(other)

        match dependency:
            case "f":
                return frechet_pbox_mul(self, other)
            case "p":
                nleft, nright = perfect_op(self, other, operator.mul)
            case "o":
                nleft, nright = opposite_op(self, other, operator.mul)
            case "i":
                nleft, nright = independent_op(self, other, operator.mul)
        return Staircase(left=nleft, right=nright)

    def div(self, other, dependency="f"):

        if self.straddles_zero():
            warnings.warn(
                "Division of a pbox straddling zero needs attention", UserWarning
            )

        if dependency == "o":
            dependency = "p"
        elif dependency == "p":
            dependency = "o"

        return self.mul(1 / other, dependency)

    def pow(self, other, dependency="f"):
        """Exponentiation of uncertain numbers with the defined dependency dependency

        This suggests that the exponent (i.e. `other`) can also be an uncertain number.
        """
        from .operation import frechet_op, vectorized_cartesian_op

        if isinstance(other, Number):
            if self.straddles_zero():
                from pyuncertainnumber import pba

                itvls = self.to_interval()
                response = itvls**other
                return pba.stacking(response)
            else:
                return pbox_number_ops(self, other, operator.pow)

        if is_un(other):
            other = convert_pbox(other)

        match dependency:
            case "f":
                nleft, nright = frechet_op(self, other, operator.pow)
            case "p":
                nleft = self.left**other.left
                nright = self.right**other.right
            case "o":
                nleft = self.left ** np.flip(other.right)
                nright = self.right ** np.flip(other.left)
            case "i":
                nleft = vectorized_cartesian_op(self.left, other.left, operator.pow)
                nright = vectorized_cartesian_op(self.right, other.right, operator.pow)
        nleft.sort()
        nright.sort()
        return Staircase(left=nleft, right=nright)

    # TODO: add a function which can be imported in operation.py
    def balchprod(self, other):
        """Frechet convolution of two pboxes when any of them straddles zero"""

        if self.straddles_zero() and other.straddles_zero():
            x0 = self.lo
            y0 = other.lo
            xx0 = self - x0
            yy0 = other - y0
            a = frechet_pbox_mul(xx0, yy0)
            b1, b2 = y0 * xx0, x0 * yy0
            b = classic_frechet_pbox(b1, b2, operator.add)
            return classic_frechet_pbox(a, b, operator.add) + x0 * y0
        if self.straddles_zero():
            x0 = self.lo
            xx0 = self - x0
            a = frechet_pbox_mul(xx0, other)
            b = x0 * other
            return frechet_pbox_mul(a, b)
        if other.straddles_zero():
            y0 = other.lo
            yy0 = other - y0
            a = frechet_pbox_mul(self, yy0)
            b = self * y0
            return classic_frechet_pbox(a, b, operator.add)
        return frechet_pbox_mul(self, other)


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
        return f"{base_repr}, shape={self.shape})"  # added back trailing ')'

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


def pbox_number_ops(pbox: Pbox, n: Number, f: callable):
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
    return Staircase(left=l, right=r, var=pbox.var)

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


def inspect_pbox(pbox):
    """quickly inspect a pbox object"""
    print(pbox)
    pbox.display(nuance="curve")
