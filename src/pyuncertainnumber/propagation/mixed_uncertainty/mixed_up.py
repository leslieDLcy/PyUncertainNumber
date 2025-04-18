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

note:
    - a univariate func case is considered
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
        dependency: dependency structure (e.g. vine copula or archimedean copula)
    """
    pass


def bi_imc(vars, func, method, dependency):
    """bivariate interval monte carlo

    args:
        dependency: dependency structure (regular copula)
    """
    pass


def slicing(
    vars,
):
    """independence assumption by now"""
    pass


def double_monte_carlo(
    vars,
    joint_distribution,
    epis_vars,
    n_a,
    n_e,
    func,
):
    # X in R5. (1000, 5) -> f(X)
    # samples: (n_ep, n_alea) e.g. (10, 1000)
    """
    args:
        joint_distribution,: a sampler based on joint distribution of aleatory variables
        epis_vars: epistemic variables
        n_a: number of aleatory samples
        n_e: number of epistemic samples
    """
    pass
