from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from functools import partial
import itertools
from ...pba.pbox_abc import convert_pbox
from ...pba.aggregation import stacking
from ...pba.intervals.intervalOperators import make_vec_interval
from ..epistemic_uncertainty.b2b import b2b

if TYPE_CHECKING:
    from ...pba.intervals import Interval
    from ...pba.distributions import Distribution
    from ...pba.pbox_abc import Pbox

"""leslie's implementation on mixed uncertainty propagation


design signature hint:
    - treat `vars` as the construct classes
    - share the same interface with minimal arguments set (vars, func, interval_strategy)
    - all these funcs will have the possibilities to return some verbose results
    - where these verbose results can be saved to disk using a decorator

note:
    - a univariate func case is considered
"""


def bi_imc(x, y, func, dependency=None, n_sam=100):
    """bivariate interval monte carlo

    args:
        x, y (Pbox) : Pbox
        func: callable which takes vector-type of inputs
        dependency: dependency structure (regular copula)
    """
    from scipy.stats import qmc

    # from pyuncertainnumber.pba.aggregation import stacking

    alpha = np.squeeze(qmc.LatinHypercube(d=1).random(n=n_sam))
    x_i = x.alpha_cut(alpha)
    y_i = y.alpha_cut(alpha)

    container = [func(_item) for _item in itertools.product(x_i, y_i)]
    return stacking(container)


# TODO: add vine copula
def interval_monte_carlo(
    vars: list[Interval | Distribution | Pbox],
    func: callable,
    interval_strategy,
    n_sam,
    dependency=None,
    **kwargs,
) -> Pbox:
    """
    Args:
        vars (list): list of uncertain variables
        dependency: dependency structure (e.g. vine copula or archimedean copula)
    """
    from scipy.stats import qmc

    p_vars = [convert_pbox(v) for v in vars]

    # this change when there's specified dependency structure
    alpha = np.squeeze(qmc.LatinHypercube(d=1).random(n=n_sam))
    itvs = [v.alpha_cut(alpha) for v in p_vars]

    #! func must expect 2D inputs
    # TODO add parallel logic herein
    b2b_f = partial(b2b, func=func, interval_strategy=interval_strategy, **kwargs)
    container = [b2b_f(_item) for _item in itertools.product(*itvs)]
    return stacking(container)


def slicing(
    vars: list[Distribution | Interval | Pbox],
    func,
    interval_strategy,
    n_sam=200,
    outer_approximate=False,
    dependency=None,
    **kwargs,
) -> Pbox:
    """equid-probaility discretisation and alpha cutting

    args:
        func: callable
        x: uncertain variable

    note:
        independence assumption by now
    """
    p_vars = [convert_pbox(v) for v in vars]

    # TODO: outer_approximate does not take n_sam yet. Fixed it
    if outer_approximate:
        itvs = [p.outer_approximate()[1] for p in p_vars]
    else:
        itvs = [v.discretise(n_sam) for v in p_vars]

    b2b_f = partial(b2b, func=func, interval_strategy=interval_strategy, **kwargs)
    container = [b2b_f(_item) for _item in itertools.product(*itvs)]
    return stacking(container)


def double_monte_carlo(
    joint_distribution,
    epis_vars,
    n_a,
    n_e,
    func,
    parallel=False,
) -> Pbox:
    # X in R5. (1000, 5) -> f(X)
    # samples: (n_ep, n_alea) e.g. (10, 1000)
    """
    args:
        joint_distribution: an aleatoric sampler based on joint distribution of aleatory variables (or marginal one in 1d case)
        epis_vars: epistemic variables
        n_a: number of aleatory samples
        n_e: number of epistemic samples
        parallel (Boolean): parallel processing. Only use it for heavy computation (black-box) due to overhead
    """

    # lhs sample array on epistemic variables
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
