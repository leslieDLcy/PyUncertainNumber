from typing import *
from .interval import *
from .pbox_base import Pbox
import numpy as np


# def min(x,y):
#     if x.__class__.__name__ == 'Pbox':
#         return x.min(y)
#     if y.__class__.__name__ == 'Pbox':
#         return y.min(x)
#     else:
#         raise NotImplementedError('At least one argument needs to be a Pbox')

# def max(x,y):
#     if x.__class__.__name__ == 'Pbox':
#         return x.max(y)
#     if y.__class__.__name__ == 'Pbox':
#         return y.max(x)
#     else:
#         raise NotImplementedError('At least one argument needs to be a Pbox')


def sum(*args: Union[list, tuple], method='f'):
    '''
    Allows the sum to be calculated for intervals and p-boxes

    **Parameters**:

        ``*args``: pboxes or intervals
        ``method`` (``f,i,o,p``): addition method to be used

    **Returns**:

        ``Interval | Pbox``: sum of interval or pbox objects within ``*args``

    .. note::

        If a list or tuple is given as the first argument, the elements of the list or tuple are used as arguments.
        If only one (non-list) argument is given, the argument is returned.


    '''

    if len(args) == 1:
        if isinstance(args[0], [list, tuple, np.ndarray]):
            args = args[0]
        else:
            return args[0]

    s = 0
    for o in args:
        if isinstance(o, [Interval, Pbox, Cbox]):
            s = o.add(s, method=method)
        else:
            s += o
    return s


def mean(*args: Union[list, tuple], method='f'):
    '''    
    Allows the mean to be calculated for intervals and p-boxes

    **Parameters**:

        l : list of pboxes or intervals

        method : pbox addition method to be used

    **Output**:

         ``Interval | Pbox``: mean of interval or pbox objects within ``*args``

    .. important::

        Implemented as

        >>> pba.sum(*args,method = method)/len(args)

    '''
    s = sum(*args, method=method)

    return s/len(args)


def mul(*args, method=None):
    for i, arg in enumerate(args):
        if i == 0:
            n = arg
        elif n.__class__.__name__ == 'Interval':
            if arg.__class__.__name__ == 'Interval':
                if method is None:
                    n *= arg
                elif method == 'p':
                    n = n.pmul(arg)
                elif method == 'o':
                    n = n.omul(arg)
                else:
                    raise Exception(
                        f"Method {method} unknown for Interval * Interval calculation")
            elif arg.__class__.__name__ == 'Pbox':
                n = arg.mul(n, method=method)
            else:
                n *= arg
        elif n.__class__.__name__ == 'Pbox':
            if method is None:
                n *= arg
            else:
                n = n.mul(arg, method=method)
        else:
            n *= arg
    return n


def sqrt(a):
    if a.__class__.__name__ == 'Interval':
        return Interval(np.sqrt(a.left), np.sqrt(a.right))
    elif a.__class__.__name__ == 'PBox':
        return Pbox(
            left=np.sqrt(a.left),
            right=np.sqrt(a.right),
        )
    else:

        return np.sqrt(a)
