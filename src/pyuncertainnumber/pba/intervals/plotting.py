from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

"""
--------------------------
Editted by Leslie Feb 2024 
MIT License
--------------------------
"""


def plot_intervals(x, y_i, **kwargs):
    """plot intervals vertically

    args:
        x: array-like precise values
            x-axis coordinates
        y_i: array-like Interval objects
            array of intervals
    """

    fig, ax = plt.subplots()

    def basic_plot(x, y_i, **kwargs):
        ax.plot([x, x], [y_i.hi, y_i.lo], "blue", **kwargs)
        if np.any(y_i.lo == y_i.hi):
            sc_x = x[y_i.lo == y_i.hi]
            sc_y = y_i[y_i.lo == y_i.hi].lo
            ax.scatter(sc_x, sc_y, c="blue", **kwargs)

    if len(x.shape) > 1:
        for xx, interval in zip(x, y_i):
            basic_plot([xx, xx], [interval.hi, interval.lo])
    else:
        basic_plot(x, y_i)
    return ax


def plot_lower_bound(x: ArrayLike, y_i: ArrayLike, **kwargs):
    """plot lower bound of intervals

    args:
        x (ArrayLike): x-axis coordinates
        y_i (ArrayLike): array of intervals
    """

    fig, ax = plt.subplots()
    ax.scatter(x, y_i.lo, label="lower bound", **kwargs)
    ax.legend()


# TODO: finsih the logic for when probability is given
def plot_mul_intervals(vec_interval, probability=None, ax=None):
    """plot multiple intervals horizontally in the 0-1 probability range

    args:
        vec_interval (Interval): a vector interval object
    """
    if (
        probability is None
    ):  # this will plot the intervals across the 0-1 probability axis
        from ..utils import equi_selection

        p_values = np.linspace(0, 1, 200)
        pp = equi_selection(p_values, len(vec_interval))

        if ax is None:
            fig, ax = plt.subplots()

        for i, intl in enumerate(vec_interval):  # horizontally plot the interval
            ax.plot([intl.lo, intl.hi], [pp[i], pp[i]], label=f"Interval {i+1}")
        ax.margins(x=0.1, y=0.1)
        return ax
    else:  # when probability is given
        print('unfinished function "plot_mul_intervals" with given probability')
