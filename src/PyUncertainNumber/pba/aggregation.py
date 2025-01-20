from __future__ import annotations
import numpy as np
import itertools
import numpy as np
from .operation import convert
from .intervalOperators import make_vec_interval
from .utils import weighted_ecdf, CDF_bundle, reweighting
import matplotlib.pyplot as plt
from .constructors import pbox_fromeF
from typing import TYPE_CHECKING
from .intervals import Interval
import importlib


if TYPE_CHECKING:
    from .pbox_base import Pbox
    from .interval import nInterval
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
    from .interval import Interval as nInterval
    from .ds import DempsterShafer
    from .intervals import Interval

    if isinstance(l_uns[0], nInterval | Interval | list):
        return stacking(l_uns, weights, display=display, **kwargs)
    elif isinstance(l_uns[0], Pbox):
        return mixture_pbox(l_uns, weights, display=display)
    elif isinstance(l_uns[0], DempsterShafer):
        return mixture_ds(l_uns, display=display)


def stacking(
    vec_interval: nInterval | Interval, weights, display=False, return_type="pbox"
):
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

    vec_interval = make_vec_interval(vec_interval)

    q1, p1 = weighted_ecdf(vec_interval.lo, weights)
    q2, p2 = weighted_ecdf(vec_interval.hi, weights)

    if display:
        fig, ax = plt.subplots()
        ax.step(q1, p1, marker="+", c="g", where="post")
        ax.step(q2, p2, marker="+", c="b", where="post")
        ax.plot([q1[0], q2[0]], [0, 0], c="b")
        ax.plot([q1[-1], q2[-1]], [1, 1], c="g")

    match return_type:
        case "pbox":
            return pbox_fromeF(CDF_bundle(q1, p1), CDF_bundle(q2, p2))
        case "ds":
            return DempsterShafer(intervals=vec_interval, masses=weights)
        case "bounds":
            return CDF_bundle(q1, p1), CDF_bundle(q2, p2)
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


def mixture_cdf():
    pass


def imposition(*args: Pbox | nInterval | float | int):
    """Returns the imposition/intersection of the p-boxes in *args

    args:
        - UN objects to be mixed

    returns:
        - Pbox

    note:
        - #TODO verfication needed for the base function `p1.imp(p2)`
    """

    def binary_imp(p1: Pbox, p2: Pbox) -> Pbox:
        return p1.imp(p2)

    xs = [convert(x) for x in args]
    return list(itertools.accumulate(xs, func=binary_imp))[-1]

    # p = xs[0]
    # for i in range(1, len(xs)):
    #     p = p.imp(xs[i])
    # return p


def envelope(*args: nInterval | Pbox | float) -> nInterval | Pbox:
    """
    .. _core.envelope:

    Allows the envelope to be calculated for intervals and p-boxes.

    The envelope is the smallest interval/pbox that contains all values within the arguments.

    **Parameters**:
        ``*args``: The arguments for which the envelope needs to be calculated. The arguments can be intervals, p-boxes, or floats.

    **Returns**:
        ``Pbox|Interval``: The envelope of the given arguments, which can be an interval or a p-box.

    .. error::

        ``ValueError``: If less than two arguments are given.

        ``TypeError``: If none of the arguments are intervals or p-boxes.

    """
    # Raise error if <2 arguments are given
    assert len(args) >= 2, "At least two arguments are required"

    # get the type of all arguments
    types = [arg.__class__.__name__ for arg in args]

    # check if all arguments are intervals or pboxes
    if "Interval" not in types and "Pbox" not in types:
        raise TypeError("At least one argument needs to be an Interval or Pbox")
    # check if there is a p-box in the arguments
    elif "Pbox" in types:
        # find first p-box
        i = types.index("Pbox")
        # move previous values to the end
        args = args[i:] + args[:i]

        e = args[0].env(args[1])
        for arg in args[2:]:
            e = e.env(arg)

    else:  # Intervals only

        left = np.min([arg.left if isinstance(arg, nInterval) else arg for arg in args])

        right = np.max(
            [arg.right if isinstance(arg, nInterval) else arg for arg in args]
        )

        e = nInterval(left, right)

    return e
