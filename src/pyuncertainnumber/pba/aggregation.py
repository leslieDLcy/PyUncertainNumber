from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import itertools
import numpy as np
from .intervalOperators import make_vec_interval
from .utils import weighted_ecdf, CDF_bundle, reweighting
import matplotlib.pyplot as plt
from .intervals import Interval
import importlib
import functools

from .pbox_abc import Staircase, convert_pbox

if TYPE_CHECKING:
    from .pbox_base import Pbox
    from .ds import DempsterShafer

makeUN = importlib.import_module("pyuncertainnumber.characterisation.core").makeUN

__all__ = ["stochastic_mixture", "envelope", "imposition", "stacking"]


@makeUN
def stochastic_mixture(l_uns, weights=None, display=False, **kwargs):
    """it could work for either Pbox, distribution, DS structure or Intervals

    args:
        - l_un (list): list of uncertain numbers
        - weights (list): list of weights
        - display (Boolean): boolean for plotting
    # TODO mix types later
    note:
        - currently only accepts same type objects
    """

    from .pbox_base import Pbox
    from .ds import DempsterShafer
    from .intervals import Interval

    if isinstance(l_uns[0], Interval | list):
        return stacking(l_uns, weights, display=display, **kwargs)
    elif isinstance(l_uns[0], Pbox):
        return mixture_pbox(l_uns, weights, display=display)
    elif isinstance(l_uns[0], DempsterShafer):
        return mixture_ds(l_uns, display=display)


def stacking(
    vec_interval: Interval | list[Interval],
    weights=None,
    display=False,
    ax=None,
    return_type="pbox",
) -> Pbox:
    """stochastic mixture operation of Intervals with probability masses

    args:
        - l_un (list): list of uncertain numbers
        - weights (list): list of weights
        - display (Boolean): boolean for plotting
        - return_type (str): {'pbox' or 'ds' or 'bounds'}

    return:
        - the left and right bound F in `cdf_bundlebounds` by default
        but can choose to return a p-box

    note:
        - together the interval and masses, it can be deemed that all the inputs
        required is jointly a DS structure
    """
    from .pbox_abc import Staircase

    vec_interval = make_vec_interval(vec_interval)
    q1, p1 = weighted_ecdf(vec_interval.lo, weights)
    q2, p2 = weighted_ecdf(vec_interval.hi, weights)

    if display:
        if ax is None:
            fig, ax = plt.subplots()
        ax.step(q1, p1, marker="+", c="g", where="post")
        ax.step(q2, p2, marker="+", c="b", where="post")
        ax.plot([q1[0], q2[0]], [0, 0], c="b")
        ax.plot([q1[-1], q2[-1]], [1, 1], c="g")

    cdf1 = CDF_bundle(q1, p1)
    cdf2 = CDF_bundle(q2, p2)

    match return_type:
        case "pbox":
            return Staircase.from_CDFbundle(cdf1, cdf2)
        case "ds":
            return DempsterShafer(intervals=vec_interval, masses=weights)
        case "bounds":
            return cdf1, cdf2
        case _:
            raise ValueError("return_type must be one of {'pbox', 'ds', 'bounds'}")


def mixture_pbox(l_pboxes, weights=None, display=False):

    from .pbox_base import Pbox

    if weights is None:
        N = len(l_pboxes)
        weights = np.repeat(1 / N, N)  # equal weights
    else:
        weights = np.array(weights) if not isinstance(weights, np.ndarray) else weights
        weights = weights / sum(weights)  # re-weighting

    lcdf = np.sum([p.left * w for p, w in zip(l_pboxes, weights)], axis=0)
    ucdf = np.sum([p.right * w for p, w in zip(l_pboxes, weights)], axis=0)
    pb = Pbox(left=lcdf, right=ucdf)
    if display:
        pb.display(style="band")
    return pb


def mixture_ds(l_ds, display=False):
    """mixture operation for DS structure"""

    from .ds import DempsterShafer

    intervals = np.concatenate([ds.disassemble()[0] for ds in l_ds], axis=0)
    # TODO check the duplicate intervals
    # assert sorted(intervals) == np.unique(intervals), "intervals replicate"
    masses = reweighting([ds.disassemble()[1] for ds in l_ds])
    return DempsterShafer(intervals, masses)
    # below is to return the mixture as in a pbox
    # return stacking(intervals, masses, display=display)


def imposition(l_un: list[Staircase | float | int]) -> Staircase:
    """Returns the imposition/intersection of the list of p-boxes

    args:
        - l_un (list): a list of UN objects to be mixed

    returns:
        - Pbox

    note:
        - #TODO verfication needed for the base function `p1.imp(p2)`
    """

    def binary_imp(p1, p2):
        return p1.imp(p2)

    xs = [convert_pbox(x) for x in l_un]
    return functools.reduce(binary_imp, xs)


def envelope(l_un):
    """calculates the envelope of uncertain number objects

    args:
        ``*args``: The components on which the envelope operation applied on.

    returns:
        ``Pbox|Interval``: The envelope of the given arguments, which can be an interval or a p-box.
    """

    def binary_env(p1, p2):
        return p1.env(p2)

    xs = [convert_pbox(x) for x in l_un]
    return functools.reduce(binary_env, xs)
