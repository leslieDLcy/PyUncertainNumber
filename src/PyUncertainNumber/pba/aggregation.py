import numpy as np
from intervals import Interval
from .interval import Interval as nInterval
from .pbox_base import Pbox
from .ds import DempsterShafer, mixture_ds
from .utils import stacking
import itertools
import numpy as np
from .operation import convert


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
    '''
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

    '''
    # Raise error if <2 arguments are given
    assert len(args) >= 2,  'At least two arguments are required'

    # get the type of all arguments
    types = [arg.__class__.__name__ for arg in args]

    # check if all arguments are intervals or pboxes
    if 'Interval' not in types and 'Pbox' not in types:
        raise TypeError(
            'At least one argument needs to be an Interval or Pbox')
    # check if there is a p-box in the arguments
    elif 'Pbox' in types:
        # find first p-box
        i = types.index('Pbox')
        # move previous values to the end
        args = args[i:] + args[:i]

        e = args[0].env(args[1])
        for arg in args[2:]:
            e = e.env(arg)

    else:  # Intervals only

        left = np.min([arg.left if isinstance(
            arg, nInterval) else arg for arg in args])

        right = np.max([arg.right if isinstance(
            arg, nInterval) else arg for arg in args])

        e = nInterval(left, right)

    return e
