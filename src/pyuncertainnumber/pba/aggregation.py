from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .intervals.intervalOperators import make_vec_interval
from .utils import weighted_ecdf, CDF_bundle, reweighting
import matplotlib.pyplot as plt
from .intervals import Interval
import functools


from .pbox_abc import Staircase, convert_pbox

if TYPE_CHECKING:
    from .pbox_abc import Pbox
    from .dss import DempsterShafer

# makeUN = importlib.import_module("pyuncertainnumber.characterisation.core").makeUN

__all__ = ["stochastic_mixture", "envelope", "imposition", "stacking", "env_ecdf"]


# TODO: if adding the decorator to make UN class
# @makeUN
# TODO: add return type argument
def stochastic_mixture(l_uns, weights=None, display=False, **kwargs):
    """it could work for either Pbox, distribution, DS structure or Intervals

    args:
        - l_un (list): list of uncertain numbers constructs
        - weights (list): list of weights
        - display (Boolean): boolean for plotting
    # TODO mix types later
    note:
        - currently only accepts same type objects
    """

    from .pbox_abc import Pbox
    from .dss import DempsterShafer
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
    **kwargs,
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
    from .dss import DempsterShafer
    from .utils import plot_two_cdf_bundle

    vec_interval = make_vec_interval(vec_interval)
    q1, p1 = weighted_ecdf(vec_interval.lo, weights)
    q2, p2 = weighted_ecdf(vec_interval.hi, weights)

    cdf1 = CDF_bundle(q1, p1)
    cdf2 = CDF_bundle(q2, p2)

    if display:
        plot_two_cdf_bundle(cdf1, cdf2, ax=ax, **kwargs)

    match return_type:
        case "pbox":
            return Staircase.from_CDFbundle(cdf1, cdf2)
        case "dss":
            return DempsterShafer(intervals=vec_interval, masses=weights)
        case "cdf":
            return cdf1, cdf2
        case _:
            raise ValueError("return_type must be one of {'pbox', 'dss', 'cdf'}")


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

    from .dss import DempsterShafer

    intervals = np.concatenate([ds.disassemble()[0] for ds in l_ds], axis=0)
    # TODO check the duplicate intervals
    # assert sorted(intervals) == np.unique(intervals), "intervals replicate"
    masses = reweighting([ds.disassemble()[1] for ds in l_ds])
    return DempsterShafer(intervals, masses)


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
    """calculates the envelope of uncertain number constructs

    args:
        l_un (array like): the components, uncertain number constructs only, on which the envelope operation applied on.

    returns:
        the envelope of the given arguments,  either a p-box or an interval.
    """

    def binary_env(p1, p2):
        return p1.env(p2)

    xs = [convert_pbox(x) for x in l_un]
    return functools.reduce(binary_env, xs)


def env_ecdf(data, ret_type="pbox", ecdf_choice="canonical"):
    """nonparametric envelope function

    arrgs:
        data (array-like): the components, uncertain number constructs only, on which the envelope operation applied on.
        ret_type (str): {'pbox' or 'cdf'}
            - default is pbox
            - cdf is the CDF bundle
        ecdf_choice (str): {'canonical' or 'staircase'}

    note:
        envelope on a set of empirical CDFs
    """
    from .utils import ecdf, weighted_ecdf

    ecdf_func = weighted_ecdf if ecdf_choice == "canonical" else ecdf

    # assume each row as a sample and eCDF
    q_list = []
    for l in range(data.shape[0]):
        dd, pp = ecdf_func(np.squeeze(data[l]))
        q_list.append(dd)

    # return the q lower bound which is the upper probability bound
    q_arr = np.array(q_list)
    l_bound = np.min(q_arr, axis=0)
    u_bound = np.max(q_arr, axis=0)

    if ret_type == "pbox":
        return Staircase(left=l_bound, right=u_bound)
    elif ret_type == "cdf":
        return CDF_bundle(l_bound, pp), CDF_bundle(u_bound, pp)
