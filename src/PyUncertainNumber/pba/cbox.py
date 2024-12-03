""" the mattered cbox modules 
Originally written by Scott in R and translated to Python also expanded functionality by Leslie
"""

import functools
import numpy as np
from PyUncertainNumber import pba
from scipy.stats import beta, t, uniform, gamma, betabinom, nbinom
from .params import Params
from intervals import Interval
import scipy
from .cbox_Leslie import cbox_from_envdists, repre_pbox, cbox_from_pseudosamples, pbox_from_pseudosamples


def interval_measurements(func):
    """ not so simple """
    @functools.wraps(func)
    def imprecise_measurements_wrapper(x, **kwargs):
        if isinstance(x, (list, np.ndarray)):
            conf_dist, params = func(x, **kwargs)
            return cbox_from_envdists(conf_dist, extre_bound_params=(params['loc'], params['scale']))

        elif isinstance(x, Interval):
            cd_lo, params_lo = func(x.lo, **kwargs)
            cd_hi, params_hi = func(x.hi, **kwargs)

            def get_interval_params():
                pass

            return cbox_from_envdists([cd_lo, cd_hi])
    return imprecise_measurements_wrapper


# ---------------------Bernoulli---------------------#

def CBbernoulli_p(x):
    n = len(x)
    k = sum(x)
    l_b_params = [k, n - k + 1]
    r_b_params = [k + 1, n - k]
    cdfs = (beta(*l_b_params), beta(*r_b_params))
    return cbox_from_envdists(cdfs, shape="beta", extre_bound_params=(l_b_params, r_b_params))

# nextvalue


def CBbernoulli(x):
    n = len(x)
    k = sum(x)
    return pba.bernoulli(np.array([k, k+1])/(n+1))


# ---------------------binomial---------------------#
# x[i] ~ binomial(N, p), for known N, x[i] is a nonnegative integer less than or equal to N
def CBbinomial_p(x, N):
    """ cbox for Bionomial parameter

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
    l_b_params = [k, n * N - k + 1]
    r_b_params = [k + 1, n * N - k]
    cdfs = (beta(*l_b_params), beta(*r_b_params))
    return cbox_from_envdists(cdfs,
                              shape="beta",
                              extre_bound_params=(l_b_params, r_b_params))


def CBbinomial(x, N):
    if isinstance(x, int):
        x = [x]
    n = len(x)
    k = sum(x)
    cdfs = (betabinom(N, k, n*N-k+1), betabinom(N, k+1, n*N-k))
    return repre_pbox(cdfs, shape="betanomial")


# ---------------------binomialnp---------------------#
# TODO not done yet
# x[i] ~ binomial(N, p), for unknown N, x[i] is a nonnegative integer
# see https://sites.google.com/site/cboxbinomialnp/
def nextvalue_binomialnp(x):
    pass


def parameter_binomialnp_n(x):
    pass


def parameter_binomialnp_p(x):
    pass


# ---------------------Poisson---------------------#
# x[i] ~ Poisson(parameter), x[i] is a nonnegative integer

def CBpoisson_mean(x):
    n = len(x)
    k = sum(x)
    l_b_params = [k, 1/n]
    r_b_params = [k + 1, 1/n]
    cdfs = (gamma(*l_b_params), gamma(*r_b_params))
    return cbox_from_envdists(cdfs, shape="gamma", extre_bound_params=(l_b_params, r_b_params))


def CBpoisson(x):
    n = len(x)
    k = sum(x)

    cdfs = (nbinom(k, 1 - 1/(n+1)),
            nbinom(k+1, 1 - 1/(n+1))
            )
    return repre_pbox(cdfs, shape="nbinom")


# ---------------------exponential---------------------#
# x[i] ~ exponential(parameter), x[i] is a nonnegative integer

def CBexponential_lambda(x):
    n = len(x)
    k = sum(x)
    conf_dist = gamma(n, scale=1/k)
    return cbox_from_envdists(conf_dist, shape="gamma", extre_bound_params=(n, 1/k))


def CBexponential(x):
    n = len(x)
    k = sum(x)

    def gammaexponential(shape, rate=1, scale=None):
        if scale is None:
            scale = 1/rate
        rate = 1/scale
        # expon(scale=gamma(a=shape, scale=1/rate))
        return scipy.stats.expon.rvs(
            scale=1/scipy.stats.gamma.rvs(
                a=shape,
                scale=scale,
                size=Params.many),
            size=Params.many
        )
    mc_samples = gammaexponential(shape=n, rate=k)
    return pbox_from_pseudosamples(mc_samples)


# ---------------------normal---------------------#

# x[i] ~ normal(mu, sigma)
def CBnormal(x):
    n = len(x)
    # pop or sample std?
    def student(v): return scipy.stats.t.rvs(v, size=Params.many)
    return cbox_from_pseudosamples(np.mean(x) + np.std(x) * student(n - 1) * np.sqrt(1 + 1 / n))

# base function for precise sample x


def cboxNormalMu_base(x):
    n = len(x)
    xm = np.mean(x)
    s = np.std(x)
    conf_dist = t(n-1, loc=xm, scale=s/np.sqrt(n))  # conf_dist --> cd
    params = {'shape': 't', 'loc': xm, 'scale': (s/np.sqrt(n))}
    return conf_dist, params

    #! --- below is the nonparametric return style --- #!
    # x_support = rv.ppf(Params.p_values)
    # return x_support, params


@interval_measurements
def CBnormal_mu(x, style='analytical'):
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
        case 'analytical':
            # if isinstance(x, (list, np.ndarray)):
            #     x_sup = cboxNormalMu_base(x)
            #     return Cbox(left=x_sup, shape="t")

            # elif isinstance(x, Interval):
            #     x_sup_lo = cboxNormalMu_base(x.lo)
            #     x_sup_hi = cboxNormalMu_base(x.hi)
            #     return Cbox(left = x_sup_lo, right = x_sup_hi, shape="t")
            return cboxNormalMu_base(x)
        # TODO return a cbox object for sample-based case
        case 'samples':
            n = len(x)
            def student(v): return scipy.stats.t.rvs(v, size=Params.many)
            # pop or sample std?
            return cbox_from_pseudosamples(np.mean(x) + np.std(x) * student(n - 1) / np.sqrt(n))


def CBnormal_sigma(x):
    # TODO the analytical distribution equation?
    def chisquared(v): return (scipy.stats.chi2.rvs(v, size=Params.many))
    def inversechisquared(v): return (1/chisquared(v))
    n = len(x)
    # pop or sample var?
    pseudo_s = np.sqrt(np.var(x)*(n-1)*inversechisquared(n-1))
    return cbox_from_pseudosamples(pseudo_s)


# * ---------------------lognormal---------------------*#

# x[i] ~ lognormal(mu, sigma), x[i] is a positive value whose logarithm is distributed as normal(mu, sigma)
def CBlognormal(x):
    n = len(x)
    def student(v): return scipy.stats.t.rvs(v, size=Params.many)
    return pbox_from_pseudosamples(np.exp(np.mean(np.log(x)) + np.std(np.log(x)) * student(n - 1) * np.sqrt(1+1/n)))


def CBlognormal_mu(x):
    n = len(x)
    def student(v): return scipy.stats.t.rvs(v, size=Params.many)
    return cbox_from_pseudosamples(np.mean(np.log(x)) + np.std(np.log(x)) * student(n - 1) / np.sqrt(n))


def CBlognormal_sigma(x):
    n = len(x)
    def chisquared(v): return (scipy.stats.chi2.rvs(v, size=Params.many))
    def inversechisquared(v): return (1/chisquared(v))
    return cbox_from_pseudosamples(np.sqrt(np.var(np.log(x))*(n-1)*inversechisquared(n-1)))


# ---------------------uniform---------------------#


# x[i] ~ uniform(midpoint, width)
# x[i] ~ uniform(minimum, maximum)
def CBuniform(x):
    r = max(x)-min(x)
    w = (r/beta(len(x)-1, 2))/2
    m = (max(x)-w)+(2*w-r)*uniform(0, 1)
    return pbox_from_pseudosamples(uniform(m-w, m+w))


def CBuniform_midpoint(x):
    r = max(x)-min(x)
    w = r/beta(len(x)-1, 2)
    m = (max(x)-w/2)+(w-(max(x)-min(x)))*uniform(0, 1)
    return cbox_from_pseudosamples(m)


def CBuniform_width(x):
    r = max(x)-min(x)
    return cbox_from_pseudosamples(r/beta(len(x)-1, 2))


def CBuniform_minimum(x):
    r = max(x)-min(x)
    w = r/beta(len(x)-1, 2)
    m = (max(x)-w/2)+(w-r)*uniform(0, 1)
    return cbox_from_pseudosamples(m-w/2)


def CBuniform_maximum(x):
    r = max(x)-min(x)
    w = r/beta(len(x)-1, 2)
    m = (max(x)-w/2)+(w-r)*uniform(0, 1)
    return cbox_from_pseudosamples(m+w/2)


# ---------------------nonparametric---------------------#

# # x[i] ~ F, a continuous but unknown distribution
# # TODO arguments not confirmed yet
# def nextvalue_nonparametric(x):
#     return (np.histogram(np.concatenate((x, [np.inf])), bins='auto'),
#             np.histogram(np.concatenate((x, [-np.inf])), bins='auto'))


# def parameter_normal_meandifference(x, y):
#     return parameter_normal_mu(x) - parameter_normal_mu(y)

# TODO arguments not confirmed yet
# def parameter_nonparametric_deconvolution(x, error):
#     z = []
#     for jj in range(len(error)):
#         z.extend(x - error[jj])
#     z.sort()
#     Q = Get_Q(len(x), len(error))
#     w = Q / sum(Q)
#     return (mixture(z, w), mixture(z[:-1] + [np.inf], w))


# ---------------------helper modules---------------------#
