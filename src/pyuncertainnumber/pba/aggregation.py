from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .intervals.intervalOperators import make_vec_interval
from .utils import reweighting
from .ecdf import eCDF_bundle, get_ecdf
from .intervals import Interval
import functools
from numbers import Number
from ..decorator import UNtoUN
from .pbox_abc import Staircase, convert_pbox
from .utils import expose_functions_as_public


if TYPE_CHECKING:
    from .pbox_abc import Pbox
    from .dss import DempsterShafer


# envelope
def _envelope(*l_constructs: Pbox | DempsterShafer | Number) -> Staircase:
    """calculates the envelope of uncertain number constructs

    args:
        l_constructs (list): the components, constructs only, on which the envelope operation applied on.

    returns:
        the envelope of the given arguments,  either a p-box or an interval.
    """

    def binary_env(p1, p2):
        return p1.env(p2)

    xs = [convert_pbox(x) for x in l_constructs]
    return functools.reduce(binary_env, xs)


# imposition
def _imposition(*l_constructs: Staircase | DempsterShafer | Number) -> Staircase:
    """Returns the imposition/intersection of the list of p-boxes

    args:
        - l_constructs (list): a list of UN objects to be mixed

    returns:
        - Pbox

    note:
        - #TODO verfication needed for the base function `p1.imp(p2)`
    """

    def binary_imp(p1, p2):
        return p1.imp(p2)

    xs = [convert_pbox(x) for x in l_constructs]
    return functools.reduce(binary_imp, xs)


# mixture
def _stochastic_mixture(*l_constructs, weights=None, display=False, **kwargs):
    """it could work for either Pbox, distribution, DS structure or Intervals

    args:
        - l_constructs (list): list of constructs of uncertain number
        - weights (list): list of weights
        - display (Boolean): boolean for plotting
    # TODO mix types later
    note:
        - currently only accepts same type objects
    """

    from .pbox_abc import Pbox
    from .dss import DempsterShafer
    from .intervals import Interval

    if isinstance(l_constructs[0], Interval | list):
        return stacking(l_constructs, weights=weights, display=display, **kwargs)
    elif isinstance(l_constructs[0], Pbox):
        return mixture_pbox(*l_constructs, weights, display=display)
    elif isinstance(l_constructs[0], DempsterShafer):
        return mixture_ds(*l_constructs, display=display)


# * --------------- construct levels ----------------- * #


def stacking(
    vec_interval: Interval | list[Interval],
    *,
    weights=None,
    display=False,
    ax=None,
    return_type="pbox",
    **kwargs,
) -> Pbox:
    """stochastic mixture operation of Intervals with probability masses

    args:
        - l_constructs (list): list of constructs of uncertain numbers
        - weights (list): list of weights
        - display (Boolean): boolean for plotting
        - return_type (str): {'pbox' or 'ds' or 'bounds'}

    return:
        - the left and right bound F in `eCDF_bundlebounds` by default
        but can choose to return a p-box

    note:
        - it takes a list of intervals or a single vectorised interval, which is
        a different signature compared to the other aggregation functions.
        - together the interval and masses, it can be deemed that all the inputs
        required is jointly a DS structure
    """
    from .pbox_abc import Staircase
    from .dss import DempsterShafer
    from .ecdf import plot_two_eCDF_bundle

    vec_interval = make_vec_interval(vec_interval)
    q1, p1 = get_ecdf(vec_interval.lo, weights)
    q2, p2 = get_ecdf(vec_interval.hi, weights)

    cdf1 = eCDF_bundle(q1, p1)
    cdf2 = eCDF_bundle(q2, p2)

    if display:
        plot_two_eCDF_bundle(cdf1, cdf2, ax=ax, **kwargs)

    match return_type:
        case "pbox":
            return Staircase.from_CDFbundle(cdf1, cdf2)
        case "dss":
            return DempsterShafer(intervals=vec_interval, masses=weights)
        case "cdf":
            return cdf1, cdf2
        case _:
            raise ValueError("return_type must be one of {'pbox', 'dss', 'cdf'}")


def mixture_pbox(*l_pboxes, weights=None, display=False) -> Pbox:

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


def mixture_ds(*l_ds, display=False) -> DempsterShafer:
    """mixture operation for DS structure"""

    from .dss import DempsterShafer

    intervals = np.concatenate([ds.intervals.to_numpy() for ds in l_ds], axis=0)
    # TODO check the duplicate intervals
    # assert sorted(intervals) == np.unique(intervals), "intervals replicate"
    masses = reweighting([ds.masses for ds in l_ds])
    return DempsterShafer(intervals, masses)


def env_ecdf(data, ret_type="pbox", ecdf_choice="canonical"):
    """nonparametric envelope function

    arrgs:
        data (array): Each row represents a distribution, on which the envelope operation applied.
        ret_type (str): {'pbox' or 'cdf'}
            - default is pbox
            - cdf is the CDF bundle
        ecdf_choice (str): {'canonical' or 'staircase'}

    note:
        envelope on a set of empirical CDFs
    """
    from .ecdf import ecdf, get_ecdf

    ecdf_func = get_ecdf if ecdf_choice == "canonical" else ecdf

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
        return eCDF_bundle(l_bound, pp), eCDF_bundle(u_bound, pp)


def env_ecdf_sep(*ecdfs, ret_type="pbox", ecdf_choice="canonical"):
    """nonparametric envelope function for separate empirical CDFs"""

    data = np.array(ecdfs)
    return env_ecdf(data, ret_type=ret_type, ecdf_choice=ecdf_choice)


# exposed APIs


# envelope = UNtoUN(_envelope)

# Mapping: public API name -> private function
api_map = {
    "envelope": _envelope,
    "imposition": _imposition,
    "stochastic_mixture": _stochastic_mixture,
}

expose_functions_as_public(api_map, UNtoUN)

__all__ = list(api_map.keys())
__all__.extend(
    [
        "stacking",
        "mixture_pbox",
        "mixture_ds",
        "env_ecdf",
        "env_ecdf_sep",
    ]
)
