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


def UNtoUN(func):
    """extend constructs mapped functions to uncertain number mapped functions
    where inputs and outsuts are uncertain numbers
    """

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        from .characterisation.uncertainNumber import UncertainNumber

        # transform the constructs to uncertain numbers
        uns = [c.construct for c in args]
        p = func(*uns, **kwargs)
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


__all__ = []


def expose_public(name, wrapper):

    def decorator(func):
        globals()[name] = wrapper(func)
        __all__.append(name)
        return func

    return decorator


# def expose_dual(public_name):
#     def decorator(func):
#         globals()[public_name] = expose_un(func)  # public wrapped version
#         return func  # private unwrapped version

#     return decorator
