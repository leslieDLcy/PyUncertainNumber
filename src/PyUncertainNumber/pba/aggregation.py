import numpy as np
from intervals import Interval
from .interval import Interval as nInterval
from .pbox_base import Pbox
from .ds import DempsterShafer, mixture_ds
from .utils import stacking


def stochastic_mixture(l_uns, weights=None, display=False, **kwargs):
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
        return stacking(l_uns, weights, display=display, **kwargs)
    elif isinstance(l_uns[0], Pbox):
        return mixture_pbox(l_uns, weights, display=display)
    elif isinstance(l_uns[0], DempsterShafer):
        return mixture_ds(l_uns, display=display)


def mixture_pbox(l_pboxes, weights=None, display=False):
    if weights is None:
        N = len(l_pboxes)
        weights = np.repeat(1/N, N)   # equal weights
    else:
        weights = np.array(weights) if not isinstance(
            weights, np.ndarray) else weights
        weights = weights / sum(weights)  # re-weighting

    lcdf = np.sum([p.left * w for p, w in zip(l_pboxes, weights)], axis=0)
    ucdf = np.sum([p.right * w for p, w in zip(l_pboxes, weights)], axis=0)
    pb = Pbox(left=lcdf, right=ucdf)
    if display:
        pb.display(style='band')
    return pb


def mixture_cdf():
    pass
