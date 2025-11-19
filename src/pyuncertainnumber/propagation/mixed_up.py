from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from functools import partial
import itertools
from ..pba.pbox_abc import convert_pbox
from ..pba.aggregation import stacking
from .b2b import b2b
from ..pba.dependency import Dependency
from ..pba.params import Params

if TYPE_CHECKING:
    from ..pba.intervals import Interval
    from ..pba.distributions import Distribution, JointDistribution
    from ..pba.pbox_abc import Pbox


def interval_monte_carlo(
    vars: list[Interval | Distribution | Pbox],
    func: callable,
    interval_strategy,
    n_sam: int,
    dependency: Dependency = None,
    random_state=None,
    side_effects=False,
    **kwargs,
) -> Pbox:
    """Interval Monte Carlo for propagation of pbox

    args:
        vars (list): a list of constructs

        func (callable) : response function. By default, iterable signature is expected.

        interval_strategy (str) :
            strategy for interval discretisation, options include {'direct', 'endpoints', 'subinterval'}

        n_sam (int):
            number of samples for each input

        dependency: dependency structure (e.g. vine copula or archimedean copula

        random_state: random seed for reproducibility

        side_effects (bool): whether return auxiliary outputs (side effects) during propagation
            If true, the alpha-cut samples in the uniform space will be returned as well.
            otherwise, the default is False and only the p-box is returned.


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
        >>> corre_matrix = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
        >>> de = pba.Dependency(family='gaussian', corr=corre_matrix)
        >>> mix = interval_monte_carlo(vars=[a,b,c],
        >>> ...       func=foo,
        >>> ...       n_sam=20,
        >>> ...       dependency=de,
        >>> ...       interval_strategy='direct')
    """

    vars = [convert_pbox(v) for v in vars]

    n_sam = int(n_sam) if isinstance(n_sam, float) else n_sam

    b2b_f = partial(b2b, func=func, interval_strategy=interval_strategy, **kwargs)

    if dependency is None:
        ndim = len(vars)
        dependency = Dependency(family="independence", k_dim=ndim)
        """old code for independent case only, which is driven by a cartesian product implementation"""

    prob_proxy_input = dependency.u_sample(n_sam, random_state=random_state)

    # TODO chanege alpha_cut to get outer_approximation
    container = []
    for i, row in enumerate(prob_proxy_input):
        x_domain = [
            v.alpha_cut(a) for v, a in zip(vars, row)
        ]  # yield a list of intervals

        response_y_itvl = b2b_f(x_domain)
        container.append(response_y_itvl)

    if not side_effects:
        return stacking(container)
    else:
        return stacking(container), prob_proxy_input


def slicing(
    vars: list[Distribution | Interval | Pbox],
    func,
    interval_strategy,
    n_slices,
    outer_discretisation=True,
    dependency=None,
    **kwargs,
) -> Pbox:
    """classic slicing algoritm for rigorous propagation of pbox

    args:
        vars (list): list of constructs

        func (callable) : response function

        interval_strategy (str) : strategy for interval discretisation, options include {'direct', 'endpoints', 'subinterval'}

        n_slices: number of slices for each input

        outer_discretisation (bool): whether to use outer discretisation for pbox.
            By default is True for rigorous propagation; however, alpha-cut style interval are also supported.

        dependency: dependency structure (e.g. vine copula or archimedean copula).

    tip:
        Merely independence assumption is supported by now. Other dependency structures are at beta developement now.

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

    # if outer_discretisation:
    #     itvs = [p.outer_discretisation(n_slices) for p in p_vars]
    # else:
    #     itvs = [v.discretise(n_slices) for v in p_vars]

    # if len(itvs) == 1:
    #     response_intvl = func(itvs[0])
    #     response_pbox = stacking(response_intvl)
    #     return response_pbox

    b2b_f = partial(b2b, func=func, interval_strategy=interval_strategy, **kwargs)
    # cartesian product of intervals -- original implementation but really slow and not efficient;
    # container = [b2b_f(_item) for _item in itertools.product(*itvs)]

    def make_u_sample(n, num_points):
        # 200 equally spaced points between 0 and 1
        # grid_1d = np.linspace(0.01, 0.99, num_points)

        grid_1d = np.linspace(Params.p_lboundary, Params.p_hboundary, num_points)
        # grid_1d = np.arange(Params.p_lboundary, Params.p_hboundary, 1 / num_points)

        # Create n-dimensional meshgrid
        mesh = np.meshgrid(*([grid_1d] * n), indexing="ij")

        # Stack and reshape to (num_points**n, n)
        u_sample = np.stack(mesh, axis=-1).reshape(-1, n)

        return u_sample

    prob_proxy_input = make_u_sample(n=len(vars), num_points=n_slices)

    container = []
    for i, row in enumerate(prob_proxy_input):
        x_domain = [
            v.alpha_cut(a) for v, a in zip(p_vars, row)
        ]  # yield a list of intervals

        response_y_itvl = b2b_f(x_domain)
        container.append(response_y_itvl)

    # masses = dependency.pdf(prob_proxy_input)

    # return stacking(container, weights=masses)

    return stacking(container)


def double_monte_carlo(
    joint_distribution: Distribution | JointDistribution,
    epistemic_vars: list[Interval],
    n_a: int,
    n_e: int,
    func: callable,
    side_effects=False,
    parallel=False,
) -> tuple[Pbox, list, np.ndarray]:
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


    side_effects (bool): whether return auxiliary outputs (side effects) during propagation
            If true, the alpha-cut samples in the uniform space will be returned as well.
            otherwise, the default is False and only the p-box is returned.

    return:
        If `side_effects` is True, a tuple containing the following items:
            - a p-box enveloping all the CDFs from the epistemic samples
            - a list of ECDFs for each epistemic sample
            - numpy array of shape ``(n_e+2, n_a)`` as a collection of CDFs for the response
            - the epistemic samples used

        Otherwise, just the p-box.


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
    from pyuncertainnumber import make_vec_interval, parse_bounds
    from pyuncertainnumber.pba.distributions import ECDF
    from pyuncertainnumber import envelope

    v = parse_bounds(epistemic_vars)
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
        xa_samples = joint_distribution.sample(n_a).reshape(-1, 1)

        E = np.tile(e, (n_a, 1))
        X_input = np.concatenate((xa_samples, E), axis=1)
        return func(X_input)

    p_func = partial(evaluate_func_on_e, n_a=n_a, func=func)
    container = map(p_func, epistemic_points)
    response = np.squeeze(np.stack(list(container), axis=0))  # (n_e, n_a)

    many_ecdfs = [ECDF(r) for r in response]
    env_pbox = envelope(*many_ecdfs, output_type="pbox")

    if not side_effects:
        return env_pbox
    else:
        return env_pbox, many_ecdfs, response, epistemic_points


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
