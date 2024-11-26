""" the mattered cbox modules 
Originally written by Scott in R and translated to Python also expanded functionality by Leslie
"""

import numpy as np
from PyUncertainNumber import pba
from scipy.stats import beta, t, uniform, gamma, betabinom, nbinom
from ..pbox_base import Pbox
from ..params import Params
from .cbox_Leslie import Cbox
from intervals import Interval, intervalise
import scipy
from scipy.stats import ecdf
import matplotlib.pyplot as plt

# ---------------------constructors---------------------#
def repre_cbox(rvs, shape=None, bound_params=None):
    """ transform into pbox object for cbox 
    
    args:
        rvs (list): list of `scipy.stats.rv_continuous` objects
    """
    if not isinstance(rvs, list|tuple): rvs = [rvs]
    bounds = [rv.ppf(Params.p_values) for rv in rvs]
    # if bound_params is not None: print(bound_params)
    
    return Cbox(
            *bounds,
            bound_params=bound_params,
            shape=shape,
        )

# used for nextvalue distribution which by nature is pbox
def repre_pbox(rvs, shape="beta", bound_params=None):
    """ transform into pbox object for cbox 
        
    args:
        rvs (list): list of scipy.stats.rv_continuous objects"""
    
    # x_sup
    bounds = [rv.ppf(Params.p_values) for rv in rvs]
    if bound_params is not None: print(bound_params)
    return Pbox(
            left=bounds[0],
            right=bounds[1],
            shape=shape,
        )


def pbox_from_pseudosamples(samples):
        """ a tmp constructor for pbox/cbox from approximate solution of the confidence/next value distribution 
        
        args:
            samples: the approximate Monte Carlo samples of the confidence/next value distribution

        note:
            ecdf is estimted from the samples and bridge to pbox/cbox
        """
        return Pbox(tranform_ecdf(samples, display=False))



import functools

def interval_measurements(func):
    @functools.wraps(func)
    def imprecise_measurements_wrapper(x, **kwargs):
        if isinstance(x, (list, np.ndarray)):
            conf_dist, params = func(x, **kwargs)
            return repre_cbox(conf_dist, bound_params=(params['loc'], params['scale']))
        
        elif isinstance(x, Interval):
            cd_lo, params_lo = func(x.lo, **kwargs) 
            cd_hi, params_hi = func(x.hi, **kwargs)

            def get_interval_params():
                pass

            return repre_cbox([cd_lo, cd_hi])
    return imprecise_measurements_wrapper





# ---------------------Bernoulli---------------------#

def CBbernoulli_p(x):
    n = len(x)
    k = sum(x)
    l_b_params = [k, n - k + 1]
    r_b_params = [k + 1, n - k]
    cdfs = (beta(*l_b_params), beta(*r_b_params))
    return repre_cbox(cdfs, shape="beta", bound_params=(l_b_params, r_b_params)) 

#nextvalue
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
    n = len(x)  #size
    k = sum(x)
    l_b_params = [k, n * N - k + 1]
    r_b_params = [k + 1, n * N - k]
    cdfs = (beta(*l_b_params), beta(*r_b_params))
    return repre_cbox(cdfs, 
                      shape="beta", 
                      bound_params=(l_b_params, r_b_params)) 


def CBbinomial(x, N):
    if isinstance(x, int):
        x = [x]
    n = len(x)  
    k = sum(x)
    cdfs = (betabinom(N, k,n*N-k+1), betabinom(N, k+1, n*N-k))
    return repre_pbox(cdfs , shape="betanomial") 



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
    return repre_cbox(cdfs, shape="gamma", bound_params=(l_b_params, r_b_params))  


def CBpoisson(x):
    n = len(x)
    k = sum(x)

    cdfs = (nbinom(k, 1 - 1/(n+1) ),
            nbinom(k+1, 1 - 1/(n+1) ) 
            )
    return repre_pbox(cdfs , shape="nbinom") 



# ---------------------exponential---------------------#
# x[i] ~ exponential(parameter), x[i] is a nonnegative integer

def CBexponential_lambda(x):
    n = len(x)
    k = sum(x)
    conf_dist = gamma(n, scale=1/k)
    return repre_cbox(conf_dist, shape="gamma", bound_params=(n, 1/k))


def CBexponential(x) :
    n = len(x)
    k = sum(x)
    def gammaexponential(shape,rate=1,scale=None) :
        if scale is None : scale = 1/rate
        rate = 1/scale
        #expon(scale=gamma(a=shape, scale=1/rate))
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


# base function for precise sample x
def cboxNormalMu_base(x):
    n = len(x)
    xm = np.mean(x)
    s = np.std(x)
    conf_dist = t(n-1, loc=xm, scale = s/np.sqrt(n))  # conf_dist --> cd
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
            def student(v) : return(scipy.stats.t.rvs(v,size=Params.many))
            return(np.mean(x) + np.std(x) * student(n - 1) / np.sqrt(n)) # pop or sample std?



# TODO the analytical distribution equation?
def CBnormal_sigma(x) :
    def inversechisquared(v) : return(1/chisquared(v))
    n = len(x) 
    return(np.sqrt(np.var(x)*(n-1)*inversechisquared(n-1))) # pop or sample var?


### old Leslie implementation ###
# def parameter_normal_mu(x):
#     n = len(x)
#     return np.mean(x) + np.std(x) * t.rvs(df=n-1, size=2000) / np.sqrt(1 + 1/n)

# def parameter_normal_sigma(x):
#     n = len(x)
#     return np.sqrt(
#         np.var(x) * (n - 1) / chi2.rvs(n - 1, size=2000)
#         )

# def nextvalue_normal(x):
#     n = len(x)
#     return np.mean(x) + np.std(x, ddof=1) * t.ppf(0.975, n - 1) * np.sqrt(1 + 1 / n)



# ---------------------lognormal---------------------#

# x[i] ~ lognormal(mu, sigma), x[i] is a positive value whose logarithm is distributed as normal(mu, sigma)
def CBlognormal(x) : 
    n = len(x)
    return(np.exp(np.mean(np.log(x)) + np.std(np.log(x)) * student(n - 1) * np.sqrt(1+1/n)))
def CBlognormal_mu(x) : 
    n = len(x)
    return(np.mean(np.log(x)) + np.std(np.log(x)) * student(n - 1) / np.sqrt(n))
def CBlognormal_sigma(x) : 
    n = len(x)
    return(np.sqrt(np.var(np.log(x))*(n-1)*inversechisquared(n-1)))


### old Leslie implementation ###
# def nextvalue_lognormal(x):
#     n = len(x)
#     return np.exp(np.mean(np.log(x)) + np.std(np.log(x), ddof=1) * t.ppf(0.975, n - 1) * np.sqrt(1 + 1 / n))

# def parameter_lognormal_mu(x):
#     n = len(x)
#     return np.mean(np.log(x)) + np.std(np.log(x), ddof=1) * t.ppf(0.975, n - 1) / np.sqrt(n)

# def parameter_lognormal_sigma(x):
#     n = len(x)
#     return np.sqrt(np.var(np.log(x), ddof=1) * (n - 1) / chisquare(n - 1))


# ---------------------uniform---------------------#


# x[i] ~ uniform(midpoint, width)
# x[i] ~ uniform(minimum, maximum)
def CBuniform(x) :
    r=max(x)-min(x)
    w=(r/beta(len(x)-1,2))/2
    m=(max(x)-w)+(2*w-r)*uniform(0,1); 
    return(uniform(m-w, m+w))
def CBuniform_midpoint(x) : 
    r = max(x)-min(x) 
    w = r/beta(len(x)-1,2)
    m = (max(x)-w/2)+(w-(max(x)-min(x)))*uniform(0,1)
    return(m)
def CBuniform_width(x) : 
    r = max(x)-min(x) 
    return(r/beta(len(x)-1,2))
def CBuniform_minimum(x) : 
    r=max(x)-min(x); 
    w=r/beta(len(x)-1,2)
    m=(max(x)-w/2)+(w-r)*uniform(0,1)
    return(m-w/2)
def CBuniform_maximum(x) : 
    r=max(x)-min(x) 
    w=r/beta(len(x)-1,2)
    m=(max(x)-w/2)+(w-r)*uniform(0,1)
    return(m+w/2)
   


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
#     w = Q / sum(Q)
#     return (mixture(z, w), mixture(z[:-1] + [np.inf], w))



# ---------------------helper modules---------------------#

def tranform_ecdf(s, display=False, **kwargs):
    """ plot the CDF
    
    args:
        s: sample
    """
    sth = ecdf(s)
    if display:
        fig, ax = plt.subplots()
        # ax.plot(x_support, p_values, color='g')
        ax.step(sth.cdf.quantiles, sth.cdf.probabilities, color='red', zorder=10, **kwargs)
        return sth.cdf.quantiles, ax
    else:
        return sth.cdf.quantiles


