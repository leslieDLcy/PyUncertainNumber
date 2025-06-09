"""Constructors for Dempester-Shafer structures."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .intervals.intervalOperators import make_vec_interval
from collections import namedtuple
import pyuncertainnumber.pba.aggregation as agg
from .intervals import Interval

dempstershafer_element = namedtuple("dempstershafer_element", ["interval", "mass"])
""" Named tuple for Dempster-Shafer elements.

note:
    - e.g. dempstershafer_element([0, 1], 0.5)
"""


class DempsterShafer:
    """Class for Dempester-Shafer structures.

    args:
    # TODO: add type hints
    # TODO: restructure the constructor needed
        - the `intervals` argument accepts wildcard vector intervals {list of list pairs or vec Interval};
        - masses (list): probability masses
    """

    def __init__(
        self, intervals: Interval | list[list] | np.ndarray, masses: list[float]
    ):

        self._intervals = make_vec_interval(intervals)
        self._masses = np.array(masses)

    def _create_DSstructure(self):
        return [
            dempstershafer_element(i, m) for i, m in zip(self._intervals, self._masses)
        ]

    def __repr__(self):
        return f"Dempster Shafer structure with {len(self._intervals)} focal elements"

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

    @property
    def naked_value(self):
        return np.round(np.sum(self._intervals.mid * self._masses), 3)

    def plot(self, style="box", ax=None, **kwargs):
        """for box type transform dss into a pbox and plot"""
        match style:
            case "box":
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

    @classmethod
    def from_dsElements(cls, *ds_elements: dempstershafer_element):
        """Create a Dempster-Shafer structure from a list of Dempster-Shafer elements."""

        ds_elements = list(*ds_elements)
        intervals = [elem.interval for elem in ds_elements]
        masses = [elem.mass for elem in ds_elements]
        return cls(intervals, masses)


def plot_DS_structure(
    vec_interval: list[Interval],
    masses=None,
    offset=0.3,
    ax=None,
    **kwargs,
):
    """plot the intervals in a vectorised form

    args:
        vec_interval: vectorised interval objects
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
        vec_interval: vectorised interval objects
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
