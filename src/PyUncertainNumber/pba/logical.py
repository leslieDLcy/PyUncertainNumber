from typing import *
from numbers import Number
import numpy as np

from .pbox_base import Pbox
from .interval import Interval


class Logical(Interval):
    '''
    Represents a logical value that can be either True or False or dunno ([False,True]).

    Inherits from the Interval class.

    **Attributes**:

        ``left`` (``bool``): The left endpoint of the logical value.

        ``right`` (``bool``): The right endpoint of the logical value.

    '''

    def __init__(self, left: bool, right: bool = None):
        super().__init__(left, right)

    def __bool__(self):
        if self.left == 0 and self.right == 0:
            return False
        if self.left == 1 and self.right == 1:
            return True
        else:
            print(
                'WARNING: Truth value of Logical is ambiguous, use pba.sometime or pba.always')
            return True

    def __repr__(self):
        if self.left == 0 and self.right == 0:
            return 'False'
        elif self.left == 1 and self.right == 1:
            return 'True'
        else:
            return 'Dunno [False,True]'

    __str__ = __repr__

    def __invert__(self):
        if self.left == 0 and self.right == 0:
            return Logical(True, True)
        if self.left == 1 and self.right == 1:
            return Logical(False, False)
        else:
            return self


def is_same_as(a: Union['Pbox', 'Interval'], b: Union['Pbox', 'Interval'], deep=False, exact_pbox=True):
    """
    Check if two objects of type 'Pbox' or 'Interval' are equal.

    **Parameters**:

        ``a``: The first object to be compared.

        ``b``: The second object to be compared.

        ``deep``: If True, performs a deep comparison, considering object identity. If False, performs a shallow comparison based on object attributes. Defaults to False.

        ``exact_pbox``: If True, performs a deep comparison of p-boxes, considering all attributes. If False, performs a shallow comparison of p-boxes, considering only the left and right attributes. Defaults to True.

    **Returns** ``True`` **if**:

        ``bool``: True if the objects have identical parameters. For Intervals this means that left and right are the same for both a and b. For p-boxes checks whether all p-box attributes are the same. If deep is True, checks whether the objects have the same id.

    **Examples**:

        >>> a = Interval(0, 2)
        >>> b = Interval(0, 2)
        >>> c = Interval(1, 3)
        >>> is_same_as(a, b)
        True
        >>> is_same_as(a, c)
        False

    For p-boxes:

        >>> a = pba.N([0,1],1)
        >>> b = pba.N([0,1],1)
        >>> c = pba.N([0,1],2)
        >>> is_same_as(a, b)
        True
        >>> is_same_as(a, c)
        False
        >>> e = pba.box(0,1,steps=2)
        >>> f = Pbox(left = [0,0],right=[1,1],steps=2)
        >>> is_same_as(e, f, exact_pbox = True)
        False
        >>> is_same_as(e, f, exact_pbox = False)
        True


    """
    if not isinstance(a, (Interval, Pbox)) or not isinstance(b, (Interval, Pbox)):
        return a == b

    if deep:
        if id(a) == id(b):
            return True
        else:
            return False
    else:
        if a.__class__.__name__ != b.__class__.__name__:
            return False
        elif isinstance(a, Pbox):

            if exact_pbox:
                if (
                    np.array_equal(a.left, b.left) and
                    np.array_equal(a.right, b.right)
                ):
                    return True
                else:
                    return False

            if (
                np.array_equal(a.left, b.left) and
                np.array_equal(a.right, b.right) and
                a.steps == b.steps and
                a.shape == b.shape and
                a.mean_left == b.mean_left and
                a.mean_right == b.mean_right and
                a.var_left == b.var_left and
                a.var_right == b.var_right
            ):
                return True
            else:
                return False

        elif isinstance(a, Interval):
            if (
                a.left == b.left and
                a.right == b.right
            ):
                return True
            else:
                return False


def always(logical: Union[Logical, Interval, Number, bool]) -> bool:
    """
    Checks whether the logical value is always true. i.e. Every value from one interval or p-box is always greater than any other values from another.

    This function takes either a Logical object, an interval or a float as input and checks if
    both the left and right attributes of the Logical object are True.
    If an interval is provided, it checks that both the left and right attributes of the Logical object are 1.
    If a numeric value is provided, it checks if the is equal to 1.

    **Parameters**:
        ``logical`` (``Logical``, ``Interval`` , ``Number``): An object representing a logical condition with 'left' and 'right' attributes, or a number between 0 and 1.

    **Returns**:
        ``bool``: True if both sides of the logical condition are True or if the float value is equal to 1, False otherwise.

    .. error::

        ``TypeError``: If the input is not an instance of Interval, Logical or a numeric value.

        ``ValueError``: If the input float is not between 0 and 1 or the interval contains values outside of [0,1]

    **Examples**:
        >>> a = Interval(0, 2)
        >>> b = Interval(1, 3)
        >>> c = Interval(4, 5)
        >>> always(a < b)
        False
        >>> always(a < c)
        True

    """

    if isinstance(logical, Logical):

        if logical.left and logical.right:
            return True
        else:
            return False

    elif isinstance(logical, Interval):
        if logical.left < 0 or logical.right > 1:
            raise ValueError(
                "If interval values needs to be between 0 and 1 (inclusive)")
        if logical.left == 1 and logical.right == 1:
            return True
        else:
            return False

    elif isinstance(logical, (bool, Number)):

        if logical < 0 or logical > 1:
            raise ValueError(
                "If numeric input needs to be between 0 and 1 (inclusive)")
        if logical == 1:
            return True
        else:
            False

    else:
        raise TypeError(
            "Input must be a Logical, Interval or a numeric value.")


def never(logical: Logical) -> bool:
    """
    Checks whether the logical value is always true. i.e. Every value from one interval or p-box is always less than any other values from another.

    This function takes either a Logical object, an interval or a float as input and checks if
    both the left and right attributes of the Logical object are False.
    If an interval is provided, it checks that both the left and right attributes of the Logical object are 0.
    If a numeric value is provided, it checks if the is equal to 0.

    **Parameters**:

        ``logical`` (``Logical``, ``Interval`` , ``Number``): An object representing a logical condition with 'left' and 'right' attributes, or a number between 0 and 1.

    **Returns**:

        ``bool``: True if both sides of the logical condition are True or if the float value is equal to 0, False otherwise.

    .. error::

        ``TypeError``: If the input is not an instance of Interval, Logical or a numeric value.

        ``ValueError``: If the input float is not between 0 and 1 or the interval contains values outside of [0,1]

    **Examples**:

        >>> a = Interval(0, 2)
        >>> b = Interval(1, 3)
        >>> c = Interval(4, 5)
        >>> never(a < b)
        False
        >>> never(a < c)
        True

    """

    if isinstance(logical, Logical):

        if not logical.left and not logical.right:
            return True
        else:
            return False

    elif isinstance(logical, Interval):
        if logical.left < 0 or logical.right > 1:
            raise ValueError(
                "If interval values needs to be between 0 and 1 (inclusive)")
        if logical.left == 0 and logical.right == 0:
            return True
        else:
            return False

    elif isinstance(logical, (bool, Number)):

        if logical < 0 or logical > 1:
            raise ValueError(
                "If numeric input needs to be between 0 and 1 (inclusive)")
        if logical == 0:
            return True
        else:
            False

    else:
        raise TypeError(
            "Input must be a Logical, Interval or a numeric value.")


def sometimes(logical: Logical) -> bool:
    """
    Checks whether the logical value is sometimes true. i.e. There exists one value from one interval or p-box is less than a values from another.

    This function takes either a Logical object, an interval or a float as input and checks if
    either the left and right attributes of the Logical object are True.
    If an interval is provided, it that both endpoints are not 0.
    If a numeric value is provided, it checks if the is not equal to 0.

    **Parameters**:

        ``logical`` (``Logical``, ``Interval`` , ``Number``): An object representing a logical condition with 'left' and 'right' attributes, or a number between 0 and 1.

    **Returns**:

        ``bool``: True if both sides of the logical condition are True or if the float value is equal to 0, False otherwise.

    .. error::

        ``TypeError``: If the input is not an instance of Interval, Logical or a numeric value.

        ``ValueError``: If the input float is not between 0 and 1 or the interval contains values outside of [0,1]

    **Examples**:

        >>> a = pba.Interval(0, 2)
        >>> b = pba.Interval(1, 4)
        >>> c = pba.Interval(3, 5)
        >>> pba.sometimes(a < b)
        True
        >>> pba.sometimes(a < c)
        True
        >>> pba.sometimes(c < b)
        True

    """

    if isinstance(logical, Logical):

        if not logical.left and not logical.right:
            return False
        else:
            return True

    elif isinstance(logical, Interval):
        if logical.left < 0 or logical.right > 1:
            raise ValueError(
                "If interval values needs to be between 0 and 1 (inclusive)")
        if logical.left != 0 or logical.right != 0:
            return True
        else:
            return False

    elif isinstance(logical, (bool, Number)):

        if 0 < logical <= 1:
            return True
        elif logical == 0:
            return False
        else:
            raise ValueError(
                "If numeric input needs to be between 0 and 1 (inclusive)")

    else:
        raise TypeError(
            "Input must be a Logical, Interval or a numeric value.")


def xtimes(logical: Logical) -> bool:
    """
    Checks whether the logical value is exclusively sometimes true. i.e. There exists one value from one interval or p-box is less than a values from another but it is not always the case.

    This function takes either a Logical object, an interval or a float as input and checks that the left value is False and the right value is True
    If an interval is provided, it that both endpoints are not 0 or 1.
    If a numeric value is provided, it checks if the is not equal to 0 or 1.

    **Parameters**:

        ``logical`` (``Logical``, ``Interval`` , ``Number``): An object representing a logical condition with 'left' and 'right' attributes, or a number between 0 and 1.

    **Returns**:

        ``bool``: True if both sides of the logical condition are True or if the float value is equal to 0, False otherwise.

    .. error::

        ``TypeError``: If the input is not an instance of Interval, Logical or a numeric value.

        ``ValueError``: If the input float is not between 0 and 1 or the interval contains values outside of [0,1]

    **Examples**:

        >>> a = pba.Interval(0, 2)
        >>> b = pba.Interval(2, 4)
        >>> c = pba.Interval(2.5,3.5)
        >>> pba.xtimes(a < b)
        False
        >>> pba.xtimes(a < c)
        False
        >>> pba.xtimes(c < b)
        True

    """

    if isinstance(logical, Logical):

        if not logical.left and logical.right:
            return True
        else:
            return False

    elif isinstance(logical, Interval):
        if logical.left < 0 or logical.right > 1:
            raise ValueError(
                "If interval values needs to be between 0 and 1 (inclusive)")
        if logical.left != 0 and logical.right != 1:
            return True
        else:
            return False

    elif isinstance(logical, (bool, Number)):

        if Interval(0, 1).straddles(logical, endpoints=False):
            return True
        elif logical == 0 or logical == 1:
            return False
        else:
            raise ValueError(
                "If numeric input needs to be between 0 and 1 (inclusive)")

    else:
        raise TypeError(
            "Input must be a Logical, Interval or a numeric value.")
