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
def repre_cbox(rvs, shape="beta", bound_params=None):
    """ transform into pbox object for cbox 
    
    args:
        rvs (list): list of scipy.stats.rv_continuous objects
    """
    
    # x_sup
    bounds = [rv.ppf(Params.p_values) for rv in rvs]
    # if bound_params is not None: print(bound_params)
    
    return Cbox(
            # left=bounds[0],
            # right=bounds[1],
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
    return repre_cbox(cdfs, shape="beta", 
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
    return repre_cbox([conf_dist], shape="gamma", bound_params=(n, 1/k))


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
# def CBnormal(x) : 
#     n = len(x)
#     return(np.mean(x) + np.std(x) * student(n - 1) * np.sqrt(1 + 1 / n))# pop or sample std?


# base function for precise sample x
def cboxNormalMu_base(x):
    n = len(x)
    xm = np.mean(x)
    s = np.std(x)
    rv = t(n-1, loc=xm, scale = s/np.sqrt(n))
    params = {'loc': xm, 'scale': (s/np.sqrt(n))}
    x_support = rv.ppf(Params.p_values)
    return x_support


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
            if isinstance(x, (list, np.ndarray)):
                x_sup = cboxNormalMu_base(x)
                return Cbox(left=x_sup, shape="t")
        
            elif isinstance(x, Interval):
                x_sup_lo = cboxNormalMu_base(x.lo)
                x_sup_hi = cboxNormalMu_base(x.hi)
                return Cbox(left = x_sup_lo, right = x_sup_hi, shape="t")
            
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

###############################################################################        
# Precise distribution constructors 
#
# Most of these functions should be replaced by better Python implementations. 
# These functions serve as placeholders so that the other constructors using 
# MLE, maxent, MoMM, and Bayes, etc. can be implemented and tested.  
#
# There are two problems that demand these functions be replaced.  (1) The 
# basic distribution constructors should yield p-boxes when any arguments are
# intervals (which these algorithms don't do). (2) These algorithms produce 
# distributions represented internally as collections of Monte Carlo deviates 
# rather than some semi-analytical or discrete representation used in Risk Calc.
# See the preamble to https://sites.google.com/site/confidenceboxes/software        

# def bernoulli(p) : return(np.random.uniform(size=Params.many) < p)

# # def beta(a,b) :
# #     #if (a==0) and (b==0) : return(env(np.repeat(0.0, many), np.repeat(1.0, many)))  # this is [0,1]
# #     if (a==0) and (b==0) : return(bernoulli(0.5))  # or should it be [0,1]?
# #     if (a==0) : return(np.repeat(0.0, many))
# #     if (b==0) : return(np.repeat(1.0, many))            
# #     return(scipy.stats.beta.rvs(a,b,size=Params.many))

# def beta1(m,s) : return(beta(m * (m * (1 - m) / (s**2) - 1), (m * (m * (1 - m) / (s**2) - 1)) * (1/m - 1)))

# def betabinomial2(size,v,w) : return(scipy.stats.binom.rvs(size,beta(v,w),size=Params.many))

# def betabinomial(size,v,w) : return(scipy.stats.betabinom.rvs(size,v,w,size=Params.many))

# def binomial(size,p) : return(scipy.stats.binom.rvs(size,p,size=Params.many))

# def chisquared(v) : return(scipy.stats.chi2.rvs(v,size=Params.many))

# def delta(a) : return(np.repeat(a,many))

# def exponential(rate=1,mean=None) :
#     if mean is None : mean = 1/rate
#     #rate = 1/mean
#     return(scipy.stats.expon.rvs(scale=mean,size=Params.many))

# def exponential1(mean=1) :
#     return(scipy.stats.expon.rvs(scale=mean,size=Params.many))

# def F(df1,df2) : return(scipy.stats.f.rvs(df1,df2,size=Params.many))

# def gamma(shape,rate=1,scale=None) :
#     if scale is None : scale = 1/rate
#     rate = 1/scale
#     return(scipy.stats.gamma.rvs(a=shape,scale=1/rate,size=Params.many))

# def gammaexponential(shape,rate=1,scale=None) :
#     if scale is None : scale = 1/rate
#     rate = 1/scale
#     #expon(scale=gamma(a=shape, scale=1/rate))
#     return(scipy.stats.expon.rvs(scale=1/scipy.stats.gamma.rvs(a=shape,scale=scale,size=Params.many),size=Params.many))

# def geometric(m) : return(scipy.stats.geom.rvs(m,size=Params.many))

# def gumbel(loc,scale) : return(scipy.stats.gumbel_r.rvs(loc,scale,size=Params.many))

# def inversechisquared(v) : return(1/chisquared(v))
    
# def inversegamma(shape, scale=None, rate=None) : 
#     if scale is None and not rate is None : scale = 1/rate
#     return(scipy.stats.invgamma.rvs(a=shape,scale=scale,size=Params.many))

# def laplace(a,b) :  return(scipy.stats.laplace.rvs(a,b,size=Params.many))

# def logistic(loc,scale) : return(scipy.stats.logistic.rvs(loc,scale,size=Params.many))

# def lognormal(m,s) : 
#     m2 = m**2; s2 = s**2
#     mlog = np.log(m2/np.sqrt(m2+s2))
#     slog = np.sqrt(np.log((m2+s2)/m2))
#     return(scipy.stats.lognorm.rvs(s=slog,scale=np.exp(mlog),size=Params.many))

# def lognormal2(mlog,slog) : return(scipy.stats.lognorm.rvs(s=slog,scale=np.exp(mlog),size=Params.many))

# #lognormal = function(mean=NULL, std=NULL, meanlog=NULL, stdlog=NULL, median=NULL, cv=NULL, name='', ...){
# #  if (is.null(meanlog) & !is.null(median)) meanlog = log(median)
# #  if (is.null(stdlog) & !is.null(cv)) stdlog = sqrt(log(cv^2 + 1))
# #  # lognormal(a, b) ~ lognormal2(log(a^2/sqrt(a^2+b^2)),sqrt(log((a^2+b^2)/a^2)))
# #  if (is.null(meanlog) & (!is.null(mean)) & (!is.null(std))) meanlog = log(mean^2/sqrt(mean^2+std^2))
# #  if (is.null(stdlog) & !is.null(mean) & !is.null(std)) stdlog = sqrt(log((mean^2+std^2)/mean^2))
# #  if (!is.null(meanlog) & !is.null(stdlog)) Slognormal0(meanlog,stdlog,name) else stop('not enough information to specify the lognormal distribution')
# #  }

# def loguniform_solve(m,v) :
#   def loguniform_f(a,m,v) : return(a*m*np.exp(2*(v/(m**2)+1)) + np.exp(2*a/m)*(a*m - 2*((m**2) + v)))
#   def LUgrid(aa, w) : return(left(aa)+(right(aa)-left(aa))*w/100.0)
#   aa = (m - np.sqrt(4*v), m)   # interval
#   a = m
#   ss = loguniform_f(a,m,v)
#   for j in range(4) :
#     for i in range(101) :  # 0:100 
#       a = LUgrid( aa, i)
#       s = abs(loguniform_f(a,m,v))
#       if s < ss :
#           ss = s
#           si = i 
#     a = LUgrid(aa, si)
#     aa = (LUgrid(aa, si-1), LUgrid(aa, si+1))  # interval
#   return(a)

# def loguniform(min=None, max=None, minlog=None, maxlog=None, mean=None, std=None) :
#     if (min is None) and (not (minlog is None)) : min = np.exp(minlog)
#     if (max is None) and (not (maxlog is None)) : max = np.exp(maxlog)  
#     if (max is None) and (not (mean is None)) and (not (std is None)) and (not (min is None)) : max = 2*(mean**2 +std**2)/mean - min
#     if (min is None) and (max is None) and (not (mean is None)) and (not(std is None)) :
#         min = loguniform_solve(mean,std**2)
#         max = 2*(mean**2 +std**2)/mean - min
#     return(scipy.stats.loguniform.rvs(min, max, size=Params.many))

# def loguniform1(m,s) : return(loguniform(mean=m, std=s))

# def negativebinomial(size,prob) : return(scipy.stats.nbinom.rvs(size,prob,size=Params.many))

# def normal(m,s) : return(scipy.stats.norm.rvs(m,s, size=Params.many))

# def pareto(mode, c) : return(scipy.stats.pareto.rvs(c,scale=mode,size=Params.many))

# def poisson(m) : return(scipy.stats.poisson.rvs(m,size=Params.many))

# def powerfunction(b,c) : return(scipy.stats.powerlaw.rvs(c,scale=b,size=Params.many))

# # parameterisation of rayleigh differs from that in pba.r
# def rayleigh(loc,scale) : return(scipy.stats.rayleigh.rvs(loc,scale,size=Params.many))

# def sawinconrad(min, mu, max) : # WHAT are the 'implicit constraints' doing?     
#   def sawinconradalpha01(mu) :
#       def f(alpha) : return(1/(1-1/np.exp(alpha)) - 1/alpha - mu)
#       if np.abs(mu-0.5)<0.000001 : return(0)      
#       return(uniroot(f,np.array((-500,500))))
#   def qsawinconrad(p, min, mu, max) : 
#         alpha = sawinconradalpha01((mu-min)/(max-min))
#         if np.abs(alpha)<0.000001 : return(min+(max-min)*p) 
#         else : min+(max-min)*((np.log(1+p*(np.exp(alpha)-1)))/alpha)
#   a = left(min);   b = right(max)
#   c = left(mu);    d = right(mu)
#   if c<a : c = a   # implicit constraints
#   if b<d : d = b
#   #return(qsawinconrad(np.random.uniform(size=Params.many), min, mu, max))
#   return(qsawinconrad(np.random.uniform(size=Params.many), min, mu, max))
  
# def student(v, size=Params.many) : return(scipy.stats.t.rvs(v,size=size))

# def uniform(a,b, size=Params.many) : return(scipy.stats.uniform.rvs(a, b-a, size=size)) # who parameterizes like this?!?!

# def triangular(min,mode,max, size=Params.many) : return(np.random.triangular(min, mode, max, size=size)) # cheating: uses random rather than scipy.stats

# def histogram(x) : return(x[(np.trunc(scipy.stats.uniform.rvs(size=Params.many)*len(x))).astype(int)])

# def mixture(x,w=None) :
#     if w is None : w = np.repeat(1,len(x))
#     print(Params.many)
#     r = np.sort(scipy.stats.uniform.rvs(size=Params.many))[::-1]
#     x = np.concatenate(([x[0]],x))
#     w = np.cumsum(np.concatenate(([0],w)))/sum(w)
#     u = []
#     j = len(x)-1
#     for p in r : 
#         while True :
#             if w[j] <= p : break
#             j = j - 1
#         u = np.concatenate(([x[j+1]],u))
#     return(u[np.argsort(scipy.stats.uniform.rvs(size=len(u)))])
