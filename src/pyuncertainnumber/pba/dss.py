"""Constructors for Dempester-Shafer structures."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .intervals.intervalOperators import make_vec_interval
from collections import namedtuple
import pyuncertainnumber.pba.aggregation as agg
from .intervals import Interval
from .mixins import NominalValueMixin, _PboxOpsMixin
from matplotlib.patches import Rectangle
from numpy.typing import ArrayLike

dempstershafer_element = namedtuple("dempstershafer_element", ["interval", "mass"])
""" Named tuple for Dempster-Shafer elements.

note:
    - e.g. dempstershafer_element([0, 1], 0.5)
"""


class DempsterShafer(NominalValueMixin, _PboxOpsMixin):
    """Class for Dempester-Shafer structures.

    args:

        intervals: expect wildcard vector intervals, vec-Interval; list of scalar intervals; list of list pairs; or 2D array;

        masses (ArrayLike): probability masses

    example:
        >>> from pyuncertainnumber import pba
        >>> dss = pba.DempsterShafer(intervals=[[1,5], [3,6]], masses=[0.5, 0.5])
        >>> dss.structures
        [dempstershafer_element(interval=[1.0,5.0], mass=0.5),
         dempstershafer_element(interval=[3.0,6.0], mass=0.5)]

    note:
        Dempster-Shafer structures are also called belief structures or evidence structures,
        and it can be converted to p-boxes.


        .. figure:: /_static/dss_pbox_illustration.png
            :alt: p-box and DSS illustration
            :align: center
            :width: 80%

            P-box and Dempster Shafer structure illustration.
    """

    def __init__(
        self,
        intervals: Interval | list[list] | list[Interval] | np.ndarray,
        masses: ArrayLike,
    ):

        self._intervals = make_vec_interval(intervals)
        self._masses = np.array(masses)

    def _create_DSstructure(self):
        return [
            dempstershafer_element(i, m) for i, m in zip(self._intervals, self._masses)
        ]

    def __repr__(self):
        return f"Dempster Shafer structure with {len(self._intervals)} focal elements"

    def _compute_nominal_value(self):
        return np.round(np.sum(self._intervals.mid * self._masses), 3)

    @property
    def structures(self):
        return self._create_DSstructure()

    @property
    def intervals(self):
        """Returns the Interval-typed focal elements of the Dempster-Shafer structure."""
        return self._intervals

    @property
    def focal_elements(self):
        """Returns the focal elements of the Dempster-Shafer structure."""
        return self._intervals

    @property
    def masses(self):
        return self._masses

    def plot(self, style="raw", ax=None, zorder=None, **kwargs):
        """for box type transform dss into a pbox and plot

        args:
            style (str): "raw" (default), "box", "pbox", "interval"
            edge_color (str): edge color for raw style. If None, use default red color.
        """
        if ax is None:
            fig, ax = plt.subplots()
        match style:
            case "raw" | "box":
                plot_dss_raw(
                    self.intervals.to_numpy(),
                    self.masses,
                    ax=ax,
                    zorder=zorder,
                    **kwargs,
                )
            case "pbox":
                dss_pbox = self.to_pbox()
                dss_pbox.plot(ax=ax, **kwargs)
            case "interval":
                plot_DS_structure(self.intervals, self.masses, ax=ax, **kwargs)

    def display(self, style="box", ax=None, **kwargs):
        self.plot(style=style, ax=ax, **kwargs)
        plt.show()

    def to_pbox(self):
        dss_pbox = agg.stacking(
            self.intervals,
            weights=self.masses,
            display=False,
            return_type="pbox",
        )
        return dss_pbox

    def _to_pbox(self):
        """for mixin use only"""
        return self.to_pbox()

    @classmethod
    def from_dsElements(cls, *ds_elements: dempstershafer_element):
        """Create a Dempster-Shafer structure from a list of Dempster-Shafer elements."""

        ds_elements = list(*ds_elements)
        intervals = [elem.interval for elem in ds_elements]
        masses = [elem.mass for elem in ds_elements]
        return cls(intervals, masses)


def plot_dss_raw(intervals, masses, edge_color=None, ax=None, zorder=None, **kwargs):
    """plot the Dempster-Shafer structures in a raw (boxes)form

    args:
        intervals: vec-Interval; list of scalar intervals; list of list pairs; or 2D array;
        masses (array-like): masses of the intervals
        ax: matplotlib axis object
    """

    if ax is None:
        fig, ax = plt.subplots()

    # bottoms of each slab
    bottoms = np.concatenate(([0], np.cumsum(masses)[:-1]))

    for (a, b), bottom, h in zip(intervals, bottoms, masses):
        rect = Rectangle(
            (a, bottom),
            b - a,
            h,
            facecolor="lightgray",
            edgecolor=edge_color if edge_color is not None else "red",
            linewidth=1,
            alpha=0.5,
            zorder=0 if zorder is None else zorder,
            **kwargs,
        )
        ax.add_patch(rect)

    ax.set_ylabel("Probability mass")

    # autoscale to rectangle data
    ax.autoscale_view()

    # add relative margins (x=0.05 = 5%, y=0.2 = 20%)
    ax.margins(x=0.05, y=0.05)

    ax.set_xlabel("$X$")


### below


def plot_dss_raw_reverse_axis(
    intervals, masses, ax=None, orientation="xy", invert_xaxis=True
):
    """
    Plot Dempster–Shafer structures as boxes.

    Args:
        intervals: list of (a, b) intervals
        masses: list or array of probability masses
        ax: matplotlib axis object
        orientation: "xy" (default) or "yx" (reversed; swaps X and Y axes)
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Bottoms (cumulative sum of masses)
    bottoms = np.concatenate(([0], np.cumsum(masses)[:-1]))

    if orientation == "xy":
        # --- Standard orientation (original version) ---
        for (a, b), bottom, h in zip(intervals, bottoms, masses):
            rect = Rectangle(
                (a, bottom),
                b - a,
                h,
                facecolor="lightgray",
                edgecolor="red",
                linewidth=1,
                alpha=0.5,
            )
            ax.add_patch(rect)

        ax.set_xlabel(r"$X$")
        ax.set_ylabel("Probability mass")

    elif orientation == "yx":
        # --- Reversed orientation (swap axes) ---
        for (a, b), bottom, h in zip(intervals, bottoms, masses):
            rect = Rectangle(
                (bottom, a),
                h,
                b - a,
                facecolor="lightgray",
                edgecolor="red",
                linewidth=1,
                alpha=0.5,
            )
            ax.add_patch(rect)

        # Reverse the new x-axis direction (1 → 0)
        if invert_xaxis:
            ax.invert_xaxis()

        # Move y-axis ticks and label to the right for clarity
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        ax.set_xlabel("Probability mass")
        ax.set_ylabel(r"$X$")

    else:
        raise ValueError("orientation must be 'xy' or 'yx'")

    # Autoscale and add margins
    ax.autoscale_view()
    ax.margins(x=0.05, y=0.05)

    return ax


### above


def plot_DS_structure(
    vec_interval: list[Interval] | np.ndarray | list[list],
    masses=None,
    offset=0.3,
    ax=None,
    **kwargs,
):
    """plot the Dempster-Shafer structures in intervals form

    args:
        vec_interval: vec-Interval; list of scalar intervals; list of list pairs; or 2D array;
        masses: masses of the intervals
        offset: offset for display the masses next to the intervals
    """
    vec_interval = make_vec_interval(vec_interval)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    for i, intl in enumerate(vec_interval):  # horizontally plot the interval
        ax.plot([intl.lo, intl.hi], [i, i], **kwargs)
        if masses is not None:
            ax.text(
                intl.hi + offset,
                i,
                f"{masses[i]:.2f}",
                verticalalignment="center",
                horizontalalignment="right",
            )
    ax.margins(x=0.2, y=0.1)
    ax.set_yticks([])
    return ax


def plot_DS_structure_with_labels(
    vec_interval: list[Interval],
    masses=None,
    offset=0.3,
    ax=None,
    **kwargs,
):
    """temp use: plot the intervals in a vectorised form

    args:
        vec_interval: vec-Interval; list of scalar intervals; list of list pairs; or 2D array;
        masses: masses of the intervals
        offset: offset for display the masses next to the intervals
    """
    vec_interval = make_vec_interval(vec_interval)

    expert_l = ["a", "b", "c", "d"]

    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    for i, intl in enumerate(vec_interval):  # horizontally plot the interval
        ax.plot([intl.lo, intl.hi], [i, i], label=f"expert {expert_l[i]}", **kwargs)

        if masses is not None:
            ax.text(
                intl.hi + offset,
                i,
                f"{masses[i]:.2f}",
                verticalalignment="center",
                horizontalalignment="right",
            )
    ax.margins(x=0.2, y=0.1)
    ax.set_yticks([])
    ax.legend()
    return ax
