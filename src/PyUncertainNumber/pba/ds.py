""" Constructors for Dempester-Shafer structures. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .intervalOperators import make_vec_interval
from collections import namedtuple
from .aggregation import stacking
from .interval import Interval as nInterval
from .intervals import Interval

dempstershafer_element = namedtuple("dempstershafer_element", ["interval", "mass"])
""" Named tuple for Dempster-Shafer elements.

note:
    - e.g. dempstershafer_element([0, 1], 0.5)
"""


class DempsterShafer:
    """Class for Dempester-Shafer structures.

    args:
        - the `intervals` argument accepts wildcard vector intervals {list of list pairs, Interval, pairs of nInterval};
        - masses (list): probability masses
    """

    def __init__(self, intervals, masses: list[float]):
        self._intrep = np.array(intervals)
        self._intervals = make_vec_interval(intervals)
        self._masses = np.array(masses)

    def _create_DSstructure(self):
        return [
            dempstershafer_element(i, m) for i, m in zip(self._intervals, self._masses)
        ]

    @property
    def structure(self):
        return self._create_DSstructure()

    @property
    def intervals(self):
        return self._intervals

    @property
    def masses(self):
        return self._masses

    def disassemble(
        self,
    ):
        return self._intrep, self._masses

    def display(self, style="box", **kwargs):
        intervals, masses = self.disassemble()
        match style:
            # TODO the to_pbox() interpolation is not perfect
            case "box":
                stacking(intervals, masses, display=True, return_type="pbox")
                # _ = self.to_pbox()
                # _.display(**kwargs)
            case "interval":
                plot_DS_structure(intervals, masses, **kwargs)

    def to_pbox(self):
        intervals, masses = self.disassemble()
        return stacking(intervals, masses, return_type="pbox")

    @classmethod
    def from_dsElements(cls, *ds_elements: dempstershafer_element):
        """Create a Dempster-Shafer structure from a list of Dempster-Shafer elements."""

        ds_elements = list(*ds_elements)
        intervals = [elem.interval for elem in ds_elements]
        masses = [elem.mass for elem in ds_elements]
        return cls(intervals, masses)


@mpl.rc_context({"text.usetex": True})
def plot_DS_structure(
    vec_interval: list[nInterval | Interval],
    weights=None,
    offset=0.3,
    ax=None,
    **kwargs,
):
    """plot the intervals in a vectorised form

    args:
        vec_interval: vectorised interval objects
        weights: weights of the intervals
        offset: offset for display the weights next to the intervals
    """
    vec_interval = make_vec_interval(vec_interval)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    for i, intl in enumerate(vec_interval):  # horizontally plot the interval
        ax.plot([intl.lo, intl.hi], [i, i], **kwargs)
        if weights is not None:
            ax.text(
                intl.hi + offset,
                i,
                f"{weights[i]:.2f}",
                verticalalignment="center",
                horizontalalignment="right",
            )
    ax.margins(x=0.2, y=0.1)
    ax.set_yticks([])
    return ax
