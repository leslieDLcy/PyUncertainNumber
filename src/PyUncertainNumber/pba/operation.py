

def convert(un):
    """ transform the input un into a Pbox object

    note:
        - theorically 'un' can be {Interval, DempsterShafer, Distribution, float, int}
    """

    from .pbox_base import Pbox
    from .interval import Interval as nInterval
    from .ds import DempsterShafer
    from .distributions import Distribution

    if isinstance(un, nInterval):
        return Pbox(un.left, un.right)
    elif isinstance(un, Pbox):
        return un
    elif isinstance(un, Distribution):
        return un.to_pbox()
    elif isinstance(un, DempsterShafer):
        return un.to_pbox()
    else:
        raise TypeError(f"Unable to convert {type(un)} object to Pbox")
