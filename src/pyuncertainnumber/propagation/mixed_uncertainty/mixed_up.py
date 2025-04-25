from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from functools import partial
import itertools

if TYPE_CHECKING:
    from ...pba.intervals import Interval
    from ...pba.distributions import Distribution
    from ...pba.pbox_abc import Pbox

"""leslie's implementation on mixed uncertainty propagation

#! - [ ] ongoing

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


# backup
# def bi_imc(x, y, func, dependency=None, n=200):
#     """bivariate interval monte carlo

#     args:
#         dependency: dependency structure (regular copula)
#     func: callable
#     x, y (Pbox) : Pbox
#     """
#     from scipy.stats import qmc
#     from pyuncertainnumber.pba.intervalOperators import make_vec_interval
#     from pyuncertainnumber.pba.aggregation import stacking

#     alpha = qmc.LatinHypercube(d=1).random(n=n)
#     x_i = make_vec_interval([x.alpha_cut(alpha) for p_v in alpha])
#     y_i = make_vec_interval([y.alpha_cut(p_v) for p_v in alpha])

#     container = []
#     for _item in itertools.product(x_i, y_i):
#         container.append(func(*_item))
#     arr_interval = make_vec_interval(container)
#     return stacking(arr_interval)


def bi_imc(x, y, func, dependency=None, n=200):
    """bivariate interval monte carlo

    args:
        dependency: dependency structure (regular copula)
    func: callable
    x, y (Pbox) : Pbox
    """
    from scipy.stats import qmc
    from pyuncertainnumber.pba.intervalOperators import make_vec_interval
    from pyuncertainnumber.pba.aggregation import stacking

    alpha = qmc.LatinHypercube(d=1).random(n=n)
    x_i = x.alpha_cut(alpha)
    y_i = y.alpha_cut(alpha)

    # container = []
    # for _item in itertools.product(x_i, y_i):
    #     container.append(func(*_item))

    container = [func(*_item) for _item in itertools.product(x_i, y_i)]
    arr_interval = make_vec_interval(container)
    return stacking(arr_interval)


def slicing(
    vars: list[Distribution | Interval | Pbox],
    func,
):
    """independence assumption by now"""

    from ...pba.pbox_abc import convert_pbox
    from ...pba.intervalOperators import make_vec_interval
    from ...pba.aggregation import stacking

    p_vars = [convert_pbox(v) for v in vars]

    itvs = [p.outer_approximate()[1] for p in p_vars]

    container = []
    for _item in itertools.product(itvs):
        container.append(func(*_item))

    # print(len(container))  # shall be 40_000  # checkedout
    arr_interval = make_vec_interval(container)
    return stacking(arr_interval)


def double_monte_carlo(
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

    # lhs sample on epistemic variables
    epistemic_points = epis_vars.endpoints_lhs_sample(n_e)

    def evaluate_func_on_e(e, n_a, func):
        """propagate wrt one point in the epistemic space

        args:
            e: one point in the epistemic space
            n_a: number of aleatory samples
            func: function to be evaluated

        note:
            by default, aleatory variable are put in front of the epistemic ones
        """
        xa_samples = joint_distribution.sample(n_a)

        E = np.tile(e, (n_a, 1))
        X_input = np.concatenate((xa_samples, E), axis=1)
        return func(X_input)

    p_func = partial(evaluate_func_on_e, n_a=n_a, func=func)

    container = map(p_func, epistemic_points)
    response = np.squeeze(np.stack(container, axis=0))
    # TODO : envelope CDFs into a pbox
    return response
