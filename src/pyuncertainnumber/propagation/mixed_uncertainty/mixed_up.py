from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from functools import partial
import itertools
from ...pba.pbox_abc import convert_pbox
from ...pba.aggregation import stacking
from ..epistemic_uncertainty.b2b import b2b

if TYPE_CHECKING:
    from ...pba.intervals import Interval
    from ...pba.distributions import Distribution, JointDistribution
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


# TODO: add vine copula
def interval_monte_carlo(
    vars: list[Interval | Distribution | Pbox],
    func: callable,
    interval_strategy,
    n_sam,
    dependency=None,
    **kwargs,
) -> Pbox:
    """Interval Monte Carlo for propagation of pbox

    args:
        vars (list): a list of constructs
        func (callable) : response function
        interval_strategy (str) :
            strategy for interval discretisation, options include {'direct', 'endpoints', 'subinterval'}
        n_sam (int):
            number of samples for each input
        dependency: dependency structure (e.g. vine copula or archimedean copula

    tip:
        Independence assumption by now. Dependency structure is at beta developement now.

    note:
        When choosing ``interval_strategy``, "direct" requires function signature to take a list of inputs,
        whereas "subinterval" and "endpoints" require the function to have a vectorised signature.

    return:
        Pbox

    example:
        >>> from pyuncertainnumber import pba
        >>> def foo(x): return x[0] ** 3 + x[1] + x[2]
        >>> a = pba.normal([2, 3], [1])
        >>> b = pba.normal([10, 14], [1])
        >>> c = pba.normal([4, 5], [1])
        >>> mix = interval_monte_carlo(vars=[a,b,c],
        >>> ...       func=foo,
        >>> ...       n_sam=20,
        >>> ...       interval_strategy='direct')
    """
    from scipy.stats import qmc

    p_vars = [convert_pbox(v) for v in vars]

    # this change when there's specified dependency structure
    alpha = np.squeeze(qmc.LatinHypercube(d=1).random(n=n_sam))
    itvs = [v.alpha_cut(alpha) for v in p_vars]

    # TODO add parallel logic herein
    b2b_f = partial(b2b, func=func, interval_strategy=interval_strategy, **kwargs)
    container = [b2b_f(_item) for _item in itertools.product(*itvs)]
    return stacking(container)


def slicing(
    vars: list[Distribution | Interval | Pbox],
    func,
    interval_strategy,
    n_slices,
    outer_discretisation=True,
    dependency=None,
    **kwargs,
) -> Pbox:
    """slicing algoritm for rigorous propagation of pbox

    args:
        vars (list): list of constructs
        func (callable) : response function
        interval_strategy (str) : strategy for interval discretisation, options include {'direct', 'endpoints', 'subinterval'}
        n_slices: number of slices for each input
        outer_discretisation (bool): whether to use outer discretisation for pbox.
            By default is True for rigorous propagation; however, alpha-cut style interval are also supported.
        dependency: dependency structure (e.g. vine copula or archimedean copula

    tip:
        Independence assumption by now. Dependency structure is at beta developement now.

    note:
        When choosing ``interval_strategy``, "direct" requires function signature to take a list of inputs,
        whereas "subinterval" and "endpoints" require the function to have a vectorised signature.

    return:
        Pbox

    example:
        >>> from pyuncertainnumber import pba
        >>> def foo(x): return x[0] ** 3 + x[1] + x[2]
        >>> a = pba.normal([2, 3], [1])
        >>> b = pba.normal([10, 14], [1])
        >>> c = pba.normal([4, 5], [1])
        >>> mix = slicing(vars=[a,b,c],
        >>> ...       func=foo,
        >>> ...       n_slices=20,
        >>> ...       interval_strategy='direct')

    """
    p_vars = [convert_pbox(v) for v in vars]

    if outer_discretisation:
        itvs = [p.outer_discretisation(n_slices) for p in p_vars]
    else:
        itvs = [v.discretise(n_slices) for v in p_vars]

    if len(itvs) == 1:
        response_intvl = response_intvl = func(itvs[0])
        response_pbox = stacking(response_intvl)
        return response_pbox
    b2b_f = partial(b2b, func=func, interval_strategy=interval_strategy, **kwargs)
    container = [b2b_f(_item) for _item in itertools.product(*itvs)]
    return stacking(container)


def double_monte_carlo(
    joint_distribution: Distribution | JointDistribution,
    epistemic_vars: list[Interval],
    n_a: int,
    n_e: int,
    func: callable,
    parallel=False,
) -> Pbox:
    """Double-loop Monte Carlo or nested Monte Carlo for mixed uncertainty propagation

    args:
        joint_distribution (Distribution or JointDistribution): an aleatoric sampler based on joint distribution of aleatory variables (or marginal one in 1d case).
            A sampler is basically anything (univariate or multivariate) that has the `sample` interface whereby it can sample a given number of samples.
        epistemic_vars (list): a list epistemic variables in the form of Interval
        n_a (int): number of aleatory samples
        n_e (int): number of epistemic samples
        parallel (Boolean): parallel processing. Only use it for heavy computation (black-box) due to overhead

    hint:
        consider a function mapping f(X) -> y

        - :math:`X` in :math:`R^5` with `n_a=1000`will suggest f(1000, 5)

        - resulting sample array: with `n_e=2`, the response :math:`y` : (n_ep+2, n_a) e.g. (4, 1000)

    return:
        numpy array of shape ``(n_e+2, n_a)`` as a collection of CDFs for the response


    note:
        The result array can be interpreted as a collection of CDFs for the response function evaluated at the aleatory samples for each epistemic sample.
        One can further envelope these CDFs into a ``Pbox`` or ``UncertainNumber`` object.

    example:
        >>> from pyuncertainnumber import pba
        >>> # vectorised function signature with matrix input (2D np.ndarray)
        >>> def foo_vec(x):
        ...     return x[:, 0] ** 3 + x[:, 1] + x[:, 2] + x[:, 3]

        >>> dist_a = pba.Distribution('gaussian', (5, 1))
        >>> dist_b = pba.Distribution('uniform', (2, 3))
        >>> c = pba.Dependency('gaussian', params=0.8)
        >>> joint_dist = pba.JointDistribution(copula=c, marginals=[dist_a, dist_b])

        >>> xe1 = pba.I(1, 2)
        >>> xe2 = pba.I(3, 4)

        >>> t = double_monte_carlo(
        ...     joint_distribution=joint_dist,
        ...     epistemic_vars=[xe1, xe2],
        ...     n_a=20,
        ...     n_e=3,
        ...     func=foo_vec
        ... )
    """
    # from epistemic vars into vec interval object
    from pyuncertainnumber import make_vec_interval

    v = make_vec_interval(epistemic_vars)
    # lhs sample array on epistemic variables
    epistemic_points = v.endpoints_lhs_sample(n_e)

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
    response = np.squeeze(np.stack(list(container), axis=0))
    # TODO : envelope CDFs into a pbox
    return response


def bi_imc(x, y, func, dependency=None, n_sam=100):
    """Bivariate interval monte carlo for convenience

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
