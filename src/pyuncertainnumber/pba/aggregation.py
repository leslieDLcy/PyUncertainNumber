from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .intervals.intervalOperators import make_vec_interval
from .utils import reweighting
from .ecdf import eCDF_bundle, get_ecdf
from .intervals import Interval
import functools
from numbers import Number
from .pbox_abc import Staircase
from .operation import convert
from numpy.typing import ArrayLike
import scipy.stats as sps
from .params import Params


if TYPE_CHECKING:
    from .pbox_abc import Pbox
    from .dss import DempsterShafer
    from .distributions import Distribution
    from ..characterisation.uncertainNumber import UncertainNumber


def envelope(
    *l_uns: Pbox | DempsterShafer | Number | Interval | Distribution | UncertainNumber,
    output_type="pbox",
) -> Staircase | UncertainNumber:
    """calculates the envelope of constructs only

    args:
        l_uns (list): the components, constructs and uncertain numbers, on which the envelope operation applied on.

        output_type (str): {'pbox' or 'uncertain_number' or 'un'}
            - default is pbox

    returns:
        the envelope of the given arguments,  either a p-box or an interval.

    example:
        >>> from pyuncertainnumber import envelope
        >>> a = pba.normal(3, 1)
        >>> b = pba.uniform(5, 8)
        >>> c = pba.normal(13, 2)
        >>> t = envelope(a, b, c, output_type='pbox') # or output_type='uncertain_number'
    """

    from ..characterisation.uncertainNumber import UncertainNumber
    from .intervals.methods import is_Interval, env

    if all([is_Interval(un) for un in l_uns]):
        e = functools.reduce(env, l_uns)
        return e

    def binary_env(p1, p2):
        return p1.env(p2)

    xs = [convert(x) for x in l_uns]
    e = functools.reduce(binary_env, xs)
    if output_type == "pbox":
        return e
    elif output_type == "uncertain_number" | "un":
        return UncertainNumber.fromConstruct(e)
    else:
        raise ValueError(
            "output_type must be one of {'pbox', 'uncertain_number', 'un'}"
        )


def imposition(
    *l_uns: Pbox | DempsterShafer | Number | Interval | Distribution | UncertainNumber,
    output_type="pbox",
) -> Staircase | UncertainNumber:
    """Returns the imposition/intersection of the list of p-boxes

    args:
        l_uns (list): a list of constructs or UN objects to be mixed

        output_type (str): {'pbox' or 'uncertain_number' or 'un'}
            - default is pbox

    returns:
        - Pbox or UncertainNumber

    example:
        >>> import pyuncertainnumber as pun
        >>> from pyuncertainnumber import pba
        >>> a = pba.normal([3, 7], 1)
        >>> b = pba.uniform([3,5], [6,9])
        >>> i = pun.imposition(a, b)

    """

    def binary_imp(p1, p2):
        return p1.imp(p2)

    xs = [convert(x) for x in l_uns]
    i = functools.reduce(binary_imp, xs)
    if output_type == "pbox":
        return i
    elif output_type == "uncertain_number" | "un":
        return UncertainNumber.fromConstruct(i)
    else:
        raise ValueError(
            "output_type must be one of {'pbox', 'uncertain_number', 'un'}"
        )


def stochastic_mixture(
    *l_uns: Pbox | DempsterShafer | Number | Interval | Distribution | UncertainNumber,
    weights=None,
):
    """it could work for either Pbox, distribution, DS structure or Intervals

    args:
        - l_uns (list): list of constructs or uncertain numbers
        - weights (list): list of weights

    example:
        >>> import pyuncertainnumber as pun
        >>> p = pun.stochastic_mixture([[1,3], [2,4]])
    """

    from .pbox_abc import Pbox
    from .dss import DempsterShafer
    from .intervals import Interval

    if all(isinstance(x, Interval | list) for x in l_uns):
        return stacking(l_uns, weights=weights)
    elif all(isinstance(x, Pbox) for x in l_uns):
        return mixture_pbox(*l_uns, weights)
    elif all(isinstance(x, DempsterShafer) for x in l_uns):
        return mixture_ds(*l_uns)
    else:
        converted_constructs = [convert(x) for x in l_uns]
        return mixture_pbox(*converted_constructs, weights)


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
        - vec_interval (list): list of Intervals or a vectorised Interval
        - weights (list): list of weights
        - display (Boolean): boolean for plotting
        - return_type (str): {'pbox' or 'ds' or 'bounds'}

    return:
        by default a p-box but can return the left and right bound F in `eCDF_bundlebounds`.

    note:
        - For intervals specifically.
        - it takes a list of intervals or a single vectorised interval, which is
        a different signature compared to the other aggregation functions.
        - together the interval and masses, it can be deemed that all the inputs
        required is jointly a DS structure

    example:
        >>> stacking([[1,3], [2,4]], weights=[0.5, 0.5], display=True)
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


def env_samples(data: ArrayLike, output_type="pbox", ecdf_choice="canonical"):
    """nonparametric envelope function directly from data samples

    args:

        data (ArrayLike): Each row represents a distribution, on which the envelope operation applied.

        output_type (str): {'pbox' or 'cdf'}
            default is pbox
            cdf is the CDF bundle

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

    if output_type == "pbox":
        return Staircase(left=l_bound, right=u_bound)
    elif output_type == "cdf":
        return eCDF_bundle(l_bound, pp), eCDF_bundle(u_bound, pp)


def env_ecdf_sep(*ecdfs, output_type="pbox", ecdf_choice="canonical"):
    """nonparametric envelope function for separate empirical CDFs"""

    data = np.array(ecdfs)
    return env_samples(data, output_type=output_type, ecdf_choice=ecdf_choice)


"""hint:

NN output: (n_sam, 2) of tuple (mu, sigma)
--> (n_sam, quantile) via ppf function. (n_sam, 200)
"""


def env_am(n_pars: ArrayLike) -> np.ndarray:
    """bespoke function used for am metric case

    args:
        n_pars (ArrayLike): (n_sam, 2) of tuple (mu, sigma) which may be a tensor
    """
    return sps.norm.ppf(q=Params.p_values, loc=n_pars[:, 0], scale=n_pars[:, 1])


def env_pbox_am(n_mean: ArrayLike, n_std: ArrayLike) -> np.ndarray:
    """bespoke function used for am metric case

    args:
        n_mean (ArrayLike): (n_sam,) of mean values which may be a tensor
        n_std (ArrayLike): (n_sam,) of standard deviation values which may be a tensor
    """

    Q = Params.p_values[None, :]  # (1, m)
    A = n_mean[:, None]  # (n, 1)
    B = n_std[:, None]  # (n, 1)

    #  Result: shape (n, m) — each row i uses (a[i], b[i]) across all q
    quantile_array = sps.norm.ppf(q=Q, loc=A, scale=B)
    l_bound = np.min(quantile_array, axis=0)
    u_bound = np.max(quantile_array, axis=0)
    return quantile_array, Staircase(left=l_bound, right=u_bound)
