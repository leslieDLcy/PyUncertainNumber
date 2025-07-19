# * --------------------- dependency context manager ---------------------*#

from contextvars import ContextVar
from contextlib import contextmanager

# Global context variable for dependency type
_current_dependency = ContextVar("current_dependency", default="f")


@contextmanager
def dependency(dep_type: str):
    """Context manager to temporarily change arithmetic dependency ('f', 'p', 'o', or 'i')"""
    token = _current_dependency.set(dep_type)
    try:
        yield
    finally:
        _current_dependency.reset(token)


def get_current_dependency():
    return _current_dependency.get()
