import functools
from .uncertainNumber import UncertainNumber


def makeUN(func):
    """return from construct a Uncertain Number object"""

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        construct = func(*args, **kwargs)
        return UncertainNumber.fromConstruct(construct)

    return wrapper_decorator
