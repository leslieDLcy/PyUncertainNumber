from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...pba.intervals import Interval
    from ...pba.distributions import Distribution
    from ...pba.pbox_abc import Pbox

"""leslie's implementation on mixed uncertainty propagation

design signature hint:
    - treat `vars` as the construct classes
    - share the same interface with minimal arguments set (vars, func, method)
    - all these funcs will have the possibilities to return some verbose results
    - where these verbose results can be saved to disk using a decorator
"""


def interval_monte_carlo(
    vars: list[Interval | Distribution | Pbox],
    func: callable,
    method: str,
    dependency,
):
    """
    Args:
        vars (list): list of uncertain variables
        dependency: dependency structure
    """
    pass


def slicing(
    vars,
):
    pass


def double_monte_carlo(
    vars,
):
    pass
