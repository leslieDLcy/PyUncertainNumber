import functools


# * ---------------------shortcuts --------------------- *#
def makeUNPbox(func):

    from .pba.pbox_parametric import _bound_pcdf
    from .characterisation.uncertainNumber import UncertainNumber

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        family_str = func(*args, **kwargs)
        p = _bound_pcdf(family_str, *args)
        return UncertainNumber.fromConstruct(p)

    return wrapper_decorator


def constructUN(func):
    """from a construct to create a UN"""
    from .characterisation.uncertainNumber import UncertainNumber

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        p = func(*args, **kwargs)
        return UncertainNumber.fromConstruct(p)

    return wrapper_decorator


def exposeUN(func):
    """from a construct to create a UN with a choice"""

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        return_construct = kwargs.pop("return_construct", False)
        p = func(*args, **kwargs)

        if return_construct:
            return p
        from .characterisation.uncertainNumber import UncertainNumber

        return UncertainNumber.fromConstruct(p)

    return wrapper_decorator
