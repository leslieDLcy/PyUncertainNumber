""" the mattered cbox modules 
Originally written by Scott in R and translated to Python also expanded functionality by Leslie
"""

import numpy as np
from scipy.stats import beta, t, uniform, gamma, chisquare, betabinom, nbinom
from .pbox_base import Pbox
from .params import Params


def repre_cbox(cdfs, steps=Params.steps, shape="beta"):
    """ transform into pbox object for cbox """
    
    # percentiles
    
    bounds = [cdf.ppf(Params.p_values) for cdf in cdfs]
    return Pbox(
            left=bounds[0],
            right=bounds[1],
            steps=steps,
            shape=shape,
        )

# ---------------------Bernoulli---------------------#
# TODO distribution not confirmed yet
def nextvalue_bernoulli(x):
    n = len(x)
    k = np.sum(x)
    return (beta(k / (n + 1), 1), beta((k + 1) / (n + 1), 1))

def parameter_bernoulli(x):
    n = len(x)
    k = np.sum(x)
    return (beta(k, n - k + 1), beta(k + 1, n - k))

# ---------------------binomial---------------------#
def nextvalue_binomial(x, N):
    n = len(x)  
    k = np.sum(x)
    cdfs = (betabinom(N,k,n*N-k+1), betabinom(N,k+1, n*N-k))
    return repre_cbox(cdfs, steps = Params.steps , shape="betanomial") 

# TODO question: while the left/right bounds are defined by beta dist
# does this mean that the cbox is a distibutional pbox?
def parameter_binomial(x, N):
    """ cbox for Bionomial parameter

    args:
        x (list): list of values
        N (int): number of trials

    return:
        cbox: cbox object
    """
    n = len(x)  #size
    k = np.sum(x)
    cdfs = (beta(k, n * N - k + 1), beta(k + 1, n * N - k))
    return repre_cbox(cdfs, steps = Params.steps , shape="beta") 


# ---------------------binomialnp---------------------#
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

def nextvalue_poisson(x):
    n = len(x)
    k = np.sum(x)
    # TODO arguments not confirmed yet
    return (nbinom(size=k, prob=1-1/(n+1)),
            nbinom(size=k+1, prob=1-1/(n+1)))

def parameter_poisson(x):
    n = len(x)
    k = np.sum(x)
    # TODO arguments not confirmed yet
    return (gamma(k, scale=1/n), gamma(k + 1, scale=1/n))


# ---------------------exponential---------------------#
# x[i] ~ exponential(parameter), x[i] is a nonnegative integer
# TODO arguments not confirmed yet
def nextvalue_exponential(x):
    n = len(x)
    k = np.sum(x)
    return (gamma(n, scale=1/k))

def parameter_exponential(x):
    n = len(x)
    k = np.sum(x)
    return (gamma(n, scale=1/k))

def qgammaexponential(p, shape, rate=1, scale=1):
    return rate * ((1 - p) ** (-1 / shape) - 1)

def rgammaexponential(many, shape, rate=1, scale=1):
    return qgammaexponential(np.random.uniform(size=many), shape, rate, scale)


# ---------------------normal---------------------#
# ! what's the `student(n-1)` function in R? cannot find it online.
def nextvalue_normal(x):
    n = len(x)
    return np.mean(x) + np.std(x, ddof=1) * t.ppf(0.975, n - 1) * np.sqrt(1 + 1 / n)

def parameter_normal_mu(x):
    n = len(x)
    return np.mean(x) + np.std(x, ddof=1) * t.ppf(0.975, n - 1) / np.sqrt(n)

def parameter_normal_sigma(x):
    n = len(x)
    return np.sqrt(np.var(x, ddof=1) * (n - 1) / chisquare(n - 1))



# ---------------------lognormal---------------------#
# TODO arguments not confirmed yet
def nextvalue_lognormal(x):
    n = len(x)
    return np.exp(np.mean(np.log(x)) + np.std(np.log(x), ddof=1) * t.ppf(0.975, n - 1) * np.sqrt(1 + 1 / n))

def parameter_lognormal_mu(x):
    n = len(x)
    return np.mean(np.log(x)) + np.std(np.log(x), ddof=1) * t.ppf(0.975, n - 1) / np.sqrt(n)

def parameter_lognormal_sigma(x):
    n = len(x)
    return np.sqrt(np.var(np.log(x), ddof=1) * (n - 1) / chisquare(n - 1))


# ---------------------uniform---------------------#
# TODO arguments not confirmed yet
def nextvalue_uniform(x):
    n = len(x)
    w = (np.max(x) - np.min(x)) / beta(n - 1, 2)
    m = (np.max(x) - w / 2) + (w - (np.max(x) - np.min(x))) * uniform.rvs()
    return uniform.rvs(loc=m - w / 2, scale=w)

def parameter_uniform_minimum(x):
    n = len(x)
    w = (np.max(x) - np.min(x)) / beta(n - 1, 2)
    m = (np.max(x) - w / 2) + (w - (np.max(x) - np.min(x))) * uniform.rvs()
    return m - w / 2

def parameter_uniform_maximum(x):
    n = len(x)
    w = (np.max(x) - np.min(x)) / beta(n - 1, 2)
    m = (np.max(x) - w / 2) + (w - (np.max(x) - np.min(x))) * uniform.rvs()
    return m + w / 2

def parameter_uniform_width(x):
    return (np.max(x) - np.min(x)) / beta(len(x) - 1, 2)

def parameter_uniform_midpoint(x):
    w = (np.max(x) - np.min(x)) / beta(len(x) - 1, 2)
    return (np.max(x) - w / 2) + (w - (np.max(x) - np.min(x))) * uniform.rvs()




# ---------------------nonparametric---------------------#
# x[i] ~ F, a continuous but unknown distribution
# TODO arguments not confirmed yet
def nextvalue_nonparametric(x):
    return (np.histogram(np.concatenate((x, [np.inf])), bins='auto'), 
            np.histogram(np.concatenate((x, [-np.inf])), bins='auto'))

def parameter_normal_meandifference(x, y):
    return parameter_normal_mu(x) - parameter_normal_mu(y)

# TODO arguments not confirmed yet
# def parameter_nonparametric_deconvolution(x, error):
#     z = []
#     for jj in range(len(error)):
#         z.extend(x - error[jj])
#     z.sort()
#     Q = Get_Q(len(x), len(error))
#     w = Q / np.sum(Q)
#     return (mixture(z, w), mixture(z[:-1] + [np.inf], w))


