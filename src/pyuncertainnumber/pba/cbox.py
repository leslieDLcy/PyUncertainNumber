from __future__ import annotations
from typing import TYPE_CHECKING
import functools
import numpy as np
from pyuncertainnumber import pba
from scipy.stats import beta, t, gamma, betabinom, nbinom
from .params import Params
from .intervals import Interval
import scipy
from .cbox_Leslie import cbox_from_extredists, cbox_from_pseudosamples
from .constructors import pbox_from_pseudosamples
from .pbox_abc import pbox_from_extredists
import pyuncertainnumber.pba.distributions as d

if TYPE_CHECKING:
    from .cbox_Leslie import Cbox


def interval_measurements(func):
    """decorator for incorporating interval valued data"""

    # TODO not finished yet
    @functools.wraps(func)
    def imprecise_measurements_wrapper(x, **kwargs):
        if isinstance(x, (list, np.ndarray)):
            conf_dist, params = func(x, **kwargs)
            return cbox_from_extredists(
                conf_dist,
                shape="t",
                extre_bound_params=(params["loc"], params["scale"]),
            )

        elif isinstance(x, Interval):
            cd_lo, params_lo = func(x.lo, **kwargs)
            cd_hi, params_hi = func(x.hi, **kwargs)

            def get_interval_params():
                pass

            return cbox_from_extredists([cd_lo, cd_hi])

    return imprecise_measurements_wrapper


def infer_cbox(family: str, data, **args) -> Cbox:
    """top-level call signature to infer a c-box given data and family, plus rarely additional kwargs

    notes:
        - data (list): a list of data samples, e.g. [2]
        - additina kwargs such as N for binomial family

    example:
        >>> infer_cbox('binomial', data=[2], N=10)
    """
    if isinstance(named_cbox.get(family), dict):
        return {k: v(data, **args) for (k, v) in named_cbox.get(family).items()}
    return named_cbox.get(family)(data, **args)


def infer_predictive_distribution(family: str, data, **args):
    """top-level call for the next value predictive distribution"""
    return named_nextvalue.get(family)(data, **args)


# * ---------------------Bernoulli---------------------*#


def CBbernoulli_p(x):
    n = len(x)
    k = sum(x)
    l_b_params = (k, n - k + 1)
    r_b_params = (k + 1, n - k)
    cdfs = (beta(*l_b_params), beta(*r_b_params))
    return cbox_from_extredists(
        cdfs, shape="beta", extre_bound_params=[l_b_params, r_b_params]
    )


# nextvalue


def CBbernoulli(x):
    n = len(x)
    k = sum(x)
    return pba.bernoulli(np.array([k, k + 1]) / (n + 1))


# * ---------------------binomial---------------------*#
# x[i] ~ binomial(N, p), for known N, x[i] is a nonnegative integer less than or equal to N
def CBbinomial_p(x, N):
    """cbox for Bionomial parameter

    args:
        x (list or int): sample data as in a list of success or number of success or
            a single int as the number of success k
        N (int): number of trials

    note:
        x[i] ~ binomial(N, p), for unknown p, x[i] is a nonnegative integer
        but x is a int number, it suggests the number of success as `k`.

    return:
        cbox: cbox object
    """
    if isinstance(x, int):
        x = [x]
    n = len(x)  # size
    k = sum(x)
    l_b_params = (k, n * N - k + 1)
    r_b_params = (k + 1, n * N - k)
    cdfs = (beta(*l_b_params), beta(*r_b_params))
    return cbox_from_extredists(
        cdfs, shape="beta", extre_bound_params=[l_b_params, r_b_params]
    )


def CBbinomial(x, N):
    if isinstance(x, int):
        x = [x]
    n = len(x)
    k = sum(x)
    cdfs = (betabinom(N, k, n * N - k + 1), betabinom(N, k + 1, n * N - k))
    return pbox_from_extredists(cdfs, shape="betanomial")


# * ---------------------binomialnp---------------------*#
# TODO not done yet
# x[i] ~ binomial(N, p), for unknown N, x[i] is a nonnegative integer
# two parameter version (N, p)
# see https://sites.google.com/site/cboxbinomialnp/
def nextvalue_binomialnp(x):
    pass


def parameter_binomialnp_n(x):
    pass


def parameter_binomialnp_p(x):
    pass


# * ---------------------Poisson---------------------*#
# x[i] ~ Poisson(parameter), x[i] is a nonnegative integer


def CBpoisson_lambda(x):
    n = len(x)
    k = sum(x)
    l_b_params = (k, 1 / n)
    r_b_params = (k + 1, 1 / n)
    cdfs = (gamma(*l_b_params), gamma(*r_b_params))
    return cbox_from_extredists(
        cdfs, shape="gamma", extre_bound_params=[l_b_params, r_b_params]
    )


def CBpoisson(x):
    n = len(x)
    k = sum(x)

    cdfs = (nbinom(k, 1 - 1 / (n + 1)), nbinom(k + 1, 1 - 1 / (n + 1)))
    return pbox_from_extredists(cdfs, shape="nbinom")


# * ---------------------exponential---------------------*#
# x[i] ~ exponential(parameter), x[i] is a nonnegative integer


def CBexponential_lambda(x):
    n = len(x)
    k = sum(x)
    conf_dist = gamma(n, scale=1 / k)
    return cbox_from_extredists(conf_dist, shape="gamma", extre_bound_params=(n, 1 / k))


def CBexponential(x):
    n = len(x)
    k = sum(x)

    def gammaexponential(shape, rate=1, scale=None):
        if scale is None:
            scale = 1 / rate
        rate = 1 / scale
        # expon(scale=gamma(a=shape, scale=1/rate))
        return scipy.stats.expon.rvs(
            scale=1 / scipy.stats.gamma.rvs(a=shape, scale=scale, size=Params.many),
            size=Params.many,
        )

    pseudo_s = gammaexponential(shape=n, rate=k)
    return pbox_from_pseudosamples(pseudo_s)


# * ---------------------normal---------------------*#


# x[i] ~ normal(mu, sigma)
def cboxNormalMu_base(x):
    """base function for precise sample x"""
    n = len(x)
    xm = np.mean(x)
    s = np.std(x)
    conf_dist = t(n - 1, loc=xm, scale=s / np.sqrt(n))  # conf_dist --> cd
    params = {"shape": "t", "loc": xm, "scale": (s / np.sqrt(n))}
    return conf_dist, params

    #! --- below is the nonparametric return style --- #!
    # x_support = rv.ppf(Params.p_values)
    # return x_support, params


@interval_measurements
def CBnormal_mu(x, style="analytical"):
    #! note the 'style' arg has no effect due to the decorator
    """
    args:
        x: (array-like) the sample data
        style: (str) the style of the output CDF, either 'analytical' or 'samples'
        size: (int) the discritisation size.
            meaning the no of ppf in analytical style and the no of MC samples in samples style

    return:
        CDF: (array-like) the CDF of the normal distribution
    """

    match style:
        case "analytical":
            # if isinstance(x, (list, np.ndarray)):
            #     x_sup = cboxNormalMu_base(x)
            #     return Cbox(left=x_sup, shape="t")

            # elif isinstance(x, Interval):
            #     x_sup_lo = cboxNormalMu_base(x.lo)
            #     x_sup_hi = cboxNormalMu_base(x.hi)
            #     return Cbox(left = x_sup_lo, right = x_sup_hi, shape="t")
            return cboxNormalMu_base(x)
        case "samples":
            n = len(x)

            def student(v):
                return scipy.stats.t.rvs(v, size=Params.many)

            # pop or sample std?
            return cbox_from_pseudosamples(
                np.mean(x) + np.std(x) * student(n - 1) / np.sqrt(n)
            )


def CBnormal_sigma(x):
    # TODO the analytical distribution equation
    def chisquared(v):
        return scipy.stats.chi2.rvs(v, size=Params.many)

    def inversechisquared(v):
        return 1 / chisquared(v)

    n = len(x)
    # pop or sample var?
    pseudo_s = np.sqrt(np.var(x) * (n - 1) * inversechisquared(n - 1))
    return cbox_from_pseudosamples(pseudo_s)


def CBnormal(x):
    n = len(x)

    # pop or sample std?
    def student(v):
        return scipy.stats.t.rvs(v, size=Params.many)

    return pbox_from_pseudosamples(
        np.mean(x) + np.std(x) * student(n - 1) * np.sqrt(1 + 1 / n)
    )


# * ---------------------lognormal---------------------*#

# x[i] ~ lognormal(mu, sigma), x[i] is a positive value whose logarithm is distributed as normal(mu, sigma)


def CBlognormal(x):
    n = len(x)
    return pbox_from_pseudosamples(
        np.exp(
            np.mean(np.log(x))
            + np.std(np.log(x)) * d.student(n - 1) * np.sqrt(1 + 1 / n)
        )
    )


def CBlognormal_mu(x):
    n = len(x)
    return cbox_from_pseudosamples(
        np.mean(np.log(x)) + np.std(np.log(x)) * d.student(n - 1) / np.sqrt(n)
    )


def CBlognormal_sigma(x):
    n = len(x)
    return cbox_from_pseudosamples(
        np.sqrt(np.var(np.log(x)) * (n - 1) * d.inversechisquared(n - 1))
    )


# * ---------------------uniform---------------------*#


# x[i] ~ uniform(midpoint, width)
# x[i] ~ uniform(minimum, maximum)


def CBuniform_midpoint(x):

    r = max(x) - min(x)
    w = r / d.beta(len(x) - 1, 2)
    m = (max(x) - w / 2) + (w - (max(x) - min(x))) * d.uniform(0, 1)
    return cbox_from_pseudosamples(m)


def CBuniform_width(x):
    r = max(x) - min(x)
    return cbox_from_pseudosamples(r / d.beta(len(x) - 1, 2))


def CBuniform_minimum(x):
    r = max(x) - min(x)
    w = r / d.beta(len(x) - 1, 2)
    m = (max(x) - w / 2) + (w - r) * d.uniform(0, 1)
    return cbox_from_pseudosamples(m - w / 2)


def CBuniform_maximum(x):
    r = max(x) - min(x)
    w = r / d.beta(len(x) - 1, 2)
    m = (max(x) - w / 2) + (w - r) * d.uniform(0, 1)
    return cbox_from_pseudosamples(m + w / 2)


# nextvalue


def CBuniform(x):
    r = max(x) - min(x)
    w = (r / d.beta(len(x) - 1, 2)) / 2
    m = (max(x) - w) + (2 * w - r) * d.uniform(0, 1)
    return pbox_from_pseudosamples(d.uniform(m - w, m + w))


# * ---------------------nonparametric---------------------*#

# x[i] ~ F, a continuous but unknown distribution
# N.B. the infinities don't plot, but they are there


def CBnonparametric(x):
    # TODO make below native to UN
    def env(x, y):
        return np.concatenate((x, y))

    def histogram(x):
        return x[(np.trunc(scipy.stats.uniform.rvs(size=2000) * len(x))).astype(int)]

    return env(
        histogram(np.concatenate((x, [-np.inf]))),
        histogram(np.concatenate((x, [np.inf]))),
    )


# x1[i] ~ normal(mu1, sigma1), x2[j] ~ normal(mu2, sigma2), x1 and x2 are independent


def CBnormal_meandifference(x1, x2):
    return CBnormal_mu(x2) - CBnormal_mu(x1)


# x[i] = Y + error[i],  error[j] ~ F,  F unknown,  Y fixed,  x[i] and error[j] are independent
def CBnonparametric_deconvolution(x, error):  # i.e., the c-box for Y

    def Get_Q(m_in, c_in, k=None):
        if k is None:
            k = np.arange((m_in * c_in + 1))

        def Q_size_GLBL(m):
            return 1 + m + m * (m + 1) / 2 + m * (m + 1) * (m + 2) * (3 * m + 1) / 24

        def Q_size_LoCL(m, c):
            return 1 + c + m * c * (c + 1) / 2

        def Grb_Q(m_in, c_in, Q_list):
            m = max(m_in, c_in)
            c = min(m_in, c_in)
            i_min = Q_size_GLBL(m - 1) + Q_size_LoCL(m, c - 1) + 1
            return Q_list[i_min : (i_min + m * c)]

        def AddingQ(m, Q_list):
            Q_list[Q_size_GLBL(m - 1) + 1] = 1
            for c in range(m):
                i_min = Q_size_GLBL(m - 1) + Q_size_LoCL(m, c) + 1
                Q1 = np.concatenate(
                    (Grb_Q(m - 1, c + 1, Q_list), np.repeat(0, (c + 1)))
                )
                Q2 = np.concatenate((np.repeat(0, m), Grb_Q(m, c, Q_list)))
                Q_list[i_min : (i_min + m * (c + 1))] = Q1 + Q2
            return Q_list[(Q_size_GLBL(m - 1) + 1) : Q_size_GLBL(m)]

        def Bld_Q(m_top):
            Q_out = np.repeat(0, Q_size_GLBL(m_top))
            Q_out[0] = 1
            for m in range(m_top):
                Q_out[(Q_size_GLBL(m) + 1) : (Q_size_GLBL(m + 1))] = AddingQ(
                    m + 1, Q_out
                )
            return Q_out

        # body of Get_Q
        m = max(m_in, c_in)
        c = min(m_in, c_in)
        return Grb_Q(m, c, Bld_Q(m))[k + 1]

    # body of CBnonparametric_deconvolution
    z = []
    for err in error:
        z = np.append(z, [x - err])
    z.sort()
    Q = Get_Q(len(x), len(error))
    w = Q / sum(Q)
    return env(mixture(z, w), mixture(np.append(z[1:], [np.inf]), w))


# * ---------------------helper modules---------------------*#


named_cbox = {
    "bernoulli": CBbernoulli_p,
    "binomial": CBbinomial_p,
    "exponential": CBexponential_lambda,
    "expon": CBexponential_lambda,
    # 'geometric': MMgeometric,
    "lognormal": {"mu": CBlognormal_mu, "sigma": CBlognormal_sigma},
    "norm": {"mu": CBnormal_mu, "sigma": CBnormal_sigma},
    "normal": {"mu": CBnormal_mu, "sigma": CBnormal_sigma},
    "gaussian": {"mu": CBnormal_mu, "sigma": CBnormal_sigma},
    "poisson": CBpoisson_lambda,
    "uniform": {
        "midpoint": CBuniform_midpoint,
        "width": CBuniform_width,
        "minimum": CBuniform_minimum,
        "maximum": CBuniform_maximum,
    },
}


named_nextvalue = {
    "bernoulli": CBbernoulli,
    "binomial": CBbinomial,
    "exponential": CBexponential,
    "expon": CBexponential,
    "lognormal": CBlognormal,
    "norm": CBnormal,
    "normal": CBnormal,
    "gaussian": CBnormal,
    "poisson": CBpoisson,
    "uniform": CBuniform,
}
