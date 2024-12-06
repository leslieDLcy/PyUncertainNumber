import numpy as np
import matplotlib.pyplot as plt
from intervals import Interval
from .interval import Interval as nInterval
from .intervalOperators import make_vec_interval
from .utils import reweighting, cdf_bundle
from .pbox_base import Pbox
from .ds import DempsterShafer


def stochastic_mixture(l_uns, weights=None, display=False):
    """ it could work for either Pbox, distribution, DS structure or Intervals 

    args:
        - l_un (list): list of uncertain numbers
        - weights (list): list of weights
        - display (Boolean): boolean for plotting
    # TODO mix types later
    note:
        - currently only accepts same type objects
    """
    if isinstance(l_uns[0], nInterval | Interval | list):
        return stacking(l_uns, weights, display=display)
    elif isinstance(l_uns[0], Pbox):
        return mixture_pbox(l_uns, weights, display=display)
    elif isinstance(l_uns[0], DempsterShafer):
        return mixture_ds(l_uns, display=display)


def mixture_pbox(l_pboxes, weights=None, display=False):
    if weights is None:
        N = len(l_pboxes)
        weights = np.repeat(1/N, N)   # equal weights
    else:
        weights = weights / sum(weights)  # re-weighting

    lcdf = np.sum([p.left * w for p, w in zip(l_pboxes, weights)], axis=0)
    ucdf = np.sum([p.right * w for p, w in zip(l_pboxes, weights)], axis=0)
    pb = Pbox(left=lcdf, right=ucdf)
    if display:
        pb.display(style='band')
    return pb


def mixture_ds(l_ds, display=False):
    """ mixture operation for DS structure """

    intervals = np.concatenate([ds.disassemble()[0] for ds in l_ds], axis=0)
    # TODO check the duplicate intervals
    # assert sorted(intervals) == np.unique(intervals), "intervals replicate"
    masses = reweighting([ds.disassemble()[1] for ds in l_ds])
    return stacking(intervals, masses, display=display)


def mixture_cdf():
    pass


def stacking(vec_interval: list[nInterval | Interval], weights, display=False):
    """ stochastic mixture operation for DS structure and Intervals 

    args:
        - l_un (list): list of uncertain numbers
        - weights (list): list of weights
        - display (Boolean): boolean for plotting

    return:
        - the left and right bounds in respective tuples
    """

    vec_interval = make_vec_interval(vec_interval)

    q1, p1 = weighted_ecdf(vec_interval.lo, weights)
    q2, p2 = weighted_ecdf(vec_interval.hi, weights)

    if display:
        fig, ax = plt.subplots()
        ax.step(q1, p1, marker='+', c='b', where='post')
        ax.step(q2, p2, marker='+', c='g', where='post')
        ax.plot([q1[0], q2[0]], [0, 0], c='b')
        ax.plot([q1[-1], q2[-1]], [1, 1], c='g')
    return cdf_bundle(q1, p1), cdf_bundle(q2, p2)


def sorting(list1, list2):
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return list1, list2


def weighted_ecdf(s, w=None, display=False):
    """ compute the weighted ecdf from (precise) sample data 

    note:
        - Sudret eq.1
    """

    if w is None:
        # weights
        N = len(s)
        w = np.repeat(1/N, N)

    s, w = sorting(s, w)
    p = np.cumsum(w)

    # for box plotting
    q = np.insert(s, 0, s[0], axis=0)
    p = np.insert(p, 0, 0., axis=0)

    if display == True:
        fig, ax = plt.subplots()
        ax.step(q, p, marker='+', where='post')

    # return quantile and probabilities
    return q, p
