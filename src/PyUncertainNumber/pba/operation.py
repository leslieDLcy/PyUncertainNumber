from .pbox_base import Pbox
from .interval import Interval as nInterval
from .ds import DempsterShafer
import itertools
import numpy as np


def convert(un):
    """ transform the input un into a Pbox object

    note:
        - theorically 'un' can be {Interval, DempsterShafer, Distribution, float, int}
    """
    if isinstance(un, nInterval):
        return Pbox(un.left, un.right)
    elif isinstance(un, Pbox):
        return un
    elif isinstance(un, DempsterShafer):
        return un.to_pbox()
    else:
        raise TypeError(f"Unable to convert {type(un)} object to Pbox")


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
