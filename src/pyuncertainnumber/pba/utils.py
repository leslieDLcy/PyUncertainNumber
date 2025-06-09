from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .intervals.intervalOperators import wc_scalar_interval, make_vec_interval
from dataclasses import dataclass
from .intervals.number import Interval
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.legend_handler import HandlerBase


def extend_ecdf(cdf):
    """add zero and one to the ecdf

    args:
        CDF_bundle
    """
    if cdf.probabilities[0] != 0:
        cdf.probabilities = np.insert(cdf.probabilities, 0, 0)
        cdf.quantiles = np.insert(cdf.quantiles, 0, cdf.quantiles[0])
    if cdf.probabilities[-1] != 1:
        cdf.probabilities = np.append(cdf.probabilities, 1)
        cdf.quantiles = np.append(cdf.quantiles, cdf.quantiles[-1])
    return cdf


def sorting(list1, list2):
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return list1, list2


def reweighting(*masses):
    """reweight the masses to sum to 1"""
    masses = np.ravel(masses)
    return masses / masses.sum()


def uniform_reparameterisation(a, b):
    """reparameterise the uniform distribution to a, b"""
    #! incorrect in the case of Interval args
    a, b = wc_scalar_interval(a), wc_scalar_interval(b)
    return a, b - a


# TODO to test this high-performance version below
def find_nearest(array, value):
    """Find index/indices of nearest value(s) in `array` to each `value`.

    Efficient for both scalar and array inputs.
    """
    array = np.asarray(array)
    value = np.atleast_1d(value)

    # Compute distances using broadcasting
    diff = np.abs(array[None, :] - value[:, None])

    # Find index of minimum difference along axis 1
    indices = np.argmin(diff, axis=1)

    # Return scalar if input was scalar
    return indices[0] if np.isscalar(value) else indices


@mpl.rc_context({"text.usetex": True})
def plot_intervals(vec_interval: list[Interval], ax=None, **kwargs):
    """plot the intervals in a vectorised form
    args:
        vec_interval: vectorised interval objects
    """
    vec_interval = make_vec_interval(vec_interval)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    for i, intl in enumerate(vec_interval):  # horizontally plot the interval
        ax.plot([intl.lo, intl.hi], [i, i], **kwargs)
    ax.margins(x=0.1, y=0.1)
    ax.set_yticks([])
    return ax


def read_json(file_name):
    f = open(file_name)
    data = json.load(f)
    return data


def is_increasing(arr):
    """check if 'arr' is increasing"""
    return np.all(np.diff(arr) >= 0)


class NotIncreasingError(Exception):
    pass


def condensation(bound, number):
    """a joint implementation for condensation"""
    if isinstance(bound, list | tuple):
        return condensation_bounds(bound, number)
    else:
        return condensation_bound(bound, number)


def condensation_bounds(bounds, number):
    """condense the bounds of number pbox

    args:
        number (int) : the number to be reduced
        bounds (list or tuple): the left and right bound to be reduced
    """
    b = bounds[0]

    if number > len(b):
        raise ValueError("Cannot sample more elements than exist in the list.")
    if len(bounds[0]) != len(bounds[1]):
        raise Exception("steps of two bounds are different")

    indices = np.linspace(0, len(b) - 1, number, dtype=int)

    l = np.array([bounds[0][i] for i in indices])
    r = np.array([bounds[1][i] for i in indices])
    return l, r


def condensation_bound(bound, number):
    """condense the bounds of number pbox

    args:
        number (int) : the number to be reduced
        bound (array-like): either the left or right bound to be reduced
    """

    if number > len(bound):
        raise ValueError("Cannot sample more elements than exist in the list.")

    indices = np.linspace(0, len(bound) - 1, number, dtype=int)

    new_bound = np.array([bound[i] for i in indices])
    return new_bound


def smooth_condensation(bounds, number=200):

    def smooth_ecdf(V, steps):

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

    l_smooth = smooth_ecdf(bounds[0], number)
    r_smooth = smooth_ecdf(bounds[1], number)
    return l_smooth, r_smooth


def equi_selection(arr, n):
    """draw n equidistant points from the array"""
    indices = np.linspace(0, len(arr) - 1, n, dtype=int)
    selected = arr[indices]
    return selected


# --- Reuse pbox rectangle key function ---
def create_colored_edge_box(x0, y0, width, height, linewidth=1):
    verts_top = [(x0, y0 + height), (x0 + width, y0 + height)]
    verts_left = [(x0, y0), (x0, y0 + height)]
    verts_bottom = [(x0, y0), (x0 + width, y0)]
    verts_right = [(x0 + width, y0), (x0 + width, y0 + height)]

    def make_patch(verts, color):
        path = mpath.Path(verts)
        return mpatches.PathPatch(
            path, edgecolor=color, facecolor="none", linewidth=linewidth
        )

    return [
        make_patch(verts_top, "green"),
        make_patch(verts_left, "green"),
        make_patch(verts_bottom, "blue"),
        make_patch(verts_right, "blue"),
    ]


# --- Custom pbox legend handler ---
class CustomEdgeRectHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        rect_patches = create_colored_edge_box(
            xdescent, ydescent, width, height, linewidth=1
        )
        for patch in rect_patches:
            patch.set_transform(trans)
        return rect_patches


import sys


def expose_functions_as_public(mapping, wrapper):
    """expose private functions as public APIs

    args:
        mapping (dict): a dictionary containing private function names mapped to public APIs
        wrapper (callable): a function that wraps the original functions (e.g., the decorator UNtoUN)

    note:
        the decorator which wraps the original function returning Pbox into returning UN, hence making the public UN API
    """
    # Get the module that called this function
    caller_globals = sys._getframe(1).f_globals
    for name, fn in mapping.items():
        caller_globals[name] = wrapper(fn)
