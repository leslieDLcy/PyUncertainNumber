from __future__ import annotations
from typing import TYPE_CHECKING
import functools
from .uncertainNumber import UncertainNumber
from .stats import *

if TYPE_CHECKING:
    from pyuncertainnumber import Interval

""" to store core functions and decorators (underway)"""


def makeUN(func):
    """return from construct a Uncertain Number object"""

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        construct = func(*args, **kwargs)
        return UncertainNumber.fromConstruct(construct)

    return wrapper_decorator
