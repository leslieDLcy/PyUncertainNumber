# -*- coding: utf-8 -*-
''' orginal code from Scott'''

"""
A few Python algorithms for fitting precise distributions to data using

     maximum likelihood
     method of matching moments
     confidence boxes
     maximum entropy
     Bayesian inference
     maximum a posteriori
     PERT
     Fermi methods

I spent too much time on the ancillary functions that construct named probability
distributions.  I assume you already have most of those, and you'll want to swap 
out the functions I made on lines 160-328, or maybe everything on lines 22-328.

     
@author: Scott Ferson
Created starting 13:56:06 ET, Thursday, 12 November 2024
"""


###############################################################################
# Ancillary (infrastructural) functions 
###############################################################################
#
# MOST OF THE FUNCTIONS IN THIS SECTION SHOULD BE REPLACED BY YOUR OWN FUNCTIONS.
#
# These functions use a Monte Carlo assemblage of deviates, stored as a Numpy
# ndarray, to model a probability distribution, and a pair of them to model a 
# p-box.  Thus, if B is a p-box, the array B[0:many] is its left side, and the 
# array B[many:(2*many)] is its rightside.
#
# Perhaps the more serious mistake is using scipy.stats for the random deviate
# algorithms rather than numpy.random.  See https://stackoverflow.com/questions/4001577/difference-between-random-draws-from-scipy-stats-rvs-and-numpy-random
# for the difference: basically, scipy.stats is creating *distributions* from 
# which we draw random values with its rvs functions, whereas numpy.random 
# is generating random values directly, with a bit less overhead.  In essence, 
# scipy generates a random variable while numpy generates random numbers.  But
# it's not clear that we need this extra stuff.
#
# Are we using the divide-by-n formula or the formula that divides by n-1?
# The MLE and MoMM estimates both expect the population formulas for standard 
# deviation.  So, in R, we needed a correction psd = sqrt(((n-1)*sd(x)^2)/n) =
# sd(x)*sqrt(1-1/n), but this correction is not needed in Python's Numpy as
#   np.std(x)             computes the population standard deviation
#   np.std(x, ddof=1)     computes the sample standard deviation
#   np.var(x)             computes the population variance
#   np.var(x, ddof=1)     computes the sample variance
#
# Python, like R, can compose arguments to make compound distributions, e.g.,
# scipy.stats.norm.rvs(np.arange(100),2,size=100) makes a compound distribution
# from normals with increasing means.
#
# Default arguments in Python are a little bit clumsier than in R.  But it is
# possible to emulate them in Python.  For example, the Python function gg() 
# will behave like the R function g().
#
# g <- function(shape,rate=1,scale=1/rate) {rate = 1/scale; cat('rate:',rate,'  ','scale:',scale,'\n')}
# g(0)         # rate: 1    scale: 1 
# g(0,1)       # rate: 1    scale: 1 
# g(0,2)       # rate: 2    scale: 0.5 
# g(0,rate=1)  # rate: 1    scale: 1 
# g(0,rate=2)  # rate: 2    scale: 0.5 
# g(0,scale=1) # rate: 1    scale: 1 
# g(0,scale=2) # rate: 0.5  scale: 2 
#
#def gg(shape,rate=1,scale=None) :
#    if scale is None : scale = 1/rate
#    rate = 1/scale
#    print('rate:',rate,'  ','scale:',scale)
# gg(0)         # rate: 1.0    scale: 1.0
# gg(0,1)       # rate: 1.0    scale: 1.0
# gg(0,2)       # rate: 2.0    scale: 0.5
# gg(0,rate=1)  # rate: 1.0    scale: 1.0
# gg(0,rate=2)  # rate: 2.0    scale: 0.5
# gg(0,scale=1) # rate: 1.0    scale: 1
# gg(0,scale=2) # rate: 0.5    scale: 2
#

import sys
import traceback
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

many = 10000  # increase for more accuracy; decrease for speed

def stop(msg) :
    print(msg)
    print(traceback.format_exc()) # way too much: traceback.print_stack()
    sys.exit(1)

def left(x) : return(np.min(x))

def right(x) : return(np.max(x))

def env(x,y) : return(np.concatenate((x,y)))

def leftside(x) : return(x[0:many])

def rightside(x) : 
    if many < len(x) : return(x[(many):(2*many)])
    else :  return(x[0:many])

def ci(b, c=0.95, alpha=None, beta=None) :
    if alpha is None : alpha=(1-c)/2
    if beta is None : beta=1-(1-c)/2
    left = np.sort(b[0:many])[round(alpha*many)]
    if (many < len(b)) : right = np.sort(b[many:len(b)])[round(beta*many)] 
    else : right = np.sort(b[0:many])[round(beta*many)]
    return((left,right))

def pl(x,y=None) : 
    if not y is None : x = [x,y]
    plt.ylim(0, 1)
    plt.xlim(min(x), max(x))

def ecdf(d) :
    d = np.array(d)
    N = d.size
    pp= np.concatenate((np.arange(N),np.arange(1,N+1)))/N
    dd = np.concatenate((d,d))
    dd.sort()
    pp.sort()
    return dd,pp

def edf(d,c=None,lw=None,ls=None) :
    if d.size==(2*many) : # p-box
        z,p = ecdf(d[0:many])
        plt.plot(z,p,c=c,lw=lw,ls=ls)
        z,p = ecdf(d[(many):(2*many)])
        plt.plot(z,p,c=c,lw=lw,ls=ls)
    else : # distribution
        z,p = ecdf(d)
        plt.plot(z,p,c=c,lw=lw,ls=ls)
    #plt.ylabel('Cumulative probability') # just makes the graph smaller

def red(d,c=None,lw=None,ls=None) : edf(d,c='r',lw=lw,ls=ls)
def cyan(d,c=None,lw=None,ls=None) : edf(d,c='c',lw=lw,ls=ls)
def blue(d,c=None,lw=None,ls=None) : edf(d,c='b',lw=lw,ls=ls)
def green(d,c=None,lw=None,ls=None) : edf(d,c='g',lw=lw,ls=ls)
def black(d,c=None,lw=None,ls=None) : edf(d,c='k',lw=lw,ls=ls)
def yellow(d,c=None,lw=None,ls=None) : edf(d,c='y',lw=lw,ls=ls)
def orange(d,c=None,lw=None,ls=None) : edf(d,c='orange',lw=lw,ls=ls)
def purple(d,c=None,lw=None,ls=None) : edf(d,c='purple',lw=lw,ls=ls)
         
def uniroot(f,a) : 
    # https://stackoverflow.com/questions/43271440/find-a-root-of-a-function-in-a-given-range
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/optimize.html#root-finding
#    from scipy.optimize import brentq
#    return(brentq(f, min(a), max(a))) #,args=(t0)) # any function arguments beyond the varied parameter
    from scipy.optimize import fsolve
    return(fsolve(f, (min(a) + max(a))/2)) 

def zbuff(x) : return(x) # use 1/zbuff(x) if x touches zero; unnecessary with Monte Carlo distribution models



###############################################################################
#  Confidence boxes (c-boxes) for parameters and the next observable value
###############################################################################

def km(k,m) :
  # The formula env(beta(k,m+1),beta(k+1,m)) is intervalized and pared to [0,1].
  # If we're using Monte Carlo deviates to model distributions and p-boxes, 
  # we don't need Bone, Bzero, np.minimum or np.maximum 
  Bzero = 1e-6
  Bone = 1-Bzero
  if ((left(k) < 0)  or (left(m) < 0)) : stop('Improper arguments to function km')
  #if is.pbox(k) or is.pbox(m) : return(uchenna(pbox(k),pbox(m)))
  #else :
  return(np.minimum(np.maximum(env(beta(left(k),right(m)+1),beta(right(k)+1,left(m))),Bzero),Bone))

def KN(k,n) :
  # The formula env(beta(k,n-k+1),beta(k+1,max(0,n-k))) is intervalized and 
  # whittled down to [0,1].  If we're using Monte Carlo deviates to represent
  # distributions and p-boxes, we don't need Bone, Bzero, minimum or maximum 
  if ((left(k) < 0) or (right(n) < right(k))) : stop('Improper arguments to function KN')
  Bzero = 1e-6
  Bone = 1-Bzero
# return(np.minimum(np.maximum(env(beta(     k,       n -     k +1),beta(      k +1,np.maximum(0,     n -      k))) ,Bzero),Bone))
  return(np.minimum(np.maximum(env(beta(left(k),right(n)-left(k)+1),beta(right(k)+1,np.maximum(0,left(n)-right(k)))),Bzero),Bone))

def FKN(k,n) :  # binomial rate inference for trials designed with a fixed-K stopping rule
  if (left(k) < 0) or (right(n) < right(k)) : stop('Improper arguments to function KN')
  Bzero = 1e-6
  Bone = 1-Bzero
  return(np.minimum(np.maximum(env(beta(left(k),right(n)-left(k)+1),beta(right(k),np.maximum(0,left(n)-right(k)))),Bzero),Bone))

# the functionality of CBbernoulli and CBbinomial is condensed into km and KN

# x[i] ~ Bernoulli(p), x[i] is either 0 or 1
def CBbernoulli(x) : 
    n = len(x)
    k = sum(x)
    return(env(bernoulli(k/(n+1)), bernoulli((k+1)/(n+1))))
def CBbernoulli_p(x) :
    n = len(x)
    k = sum(x)
    return(env(beta(k, n-k+1), beta(k+1, n-k)))

# x[i] ~ binomial(N, p), for known N, x[i] is a nonnegative integer less than or equal to N
def CBbinomial(N,x) :
    n = len(x)
    k = sum(x)
    return(env(betabinomial(N,k,n*N-k+1),betabinomial(N,k+1, n*N-k)))
def CBbinomial_p(N,x) :
    n = len(x)
    k = sum(x)
    return(env(beta(k, n*N-k+1), beta(k+1, n*N-k)))

# x[i] ~ binomial(N, p), for unknown N, x[i] is a nonnegative integer
# see https://sites.google.com/site/cboxbinomialnp/
def CBbinomialnp(x) : stop('see https://sites.google.com/site/cboxbinomialnp/')
def CBbinomialnp_n(x) : stop('see https://sites.google.com/site/cboxbinomialnp/')
def CBbinomialnp_p(x) : stop('see https://sites.google.com/site/cboxbinomialnp/')

# x[i] ~ Poisson(mean), x[i] is a nonnegative integer
def CBpoisson(x) :
    n = len(x)
    k = sum(x)
    return(env(negativebinomial(size=k, prob=1-1/(n+1)),negativebinomial(size=k+1, prob=1-1/(n+1))))
def CBpoisson_mean(x) :
    n = len(x)
    k = sum(x)
    return(env(gamma(shape=k, rate=n),gamma(shape=k+1, rate=n)))

# x[i] ~ exponential(mean), x[i] is a nonnegative integer
def CBexponential(x) :
    n = len(x)
    k = sum(x)
    return(gammaexponential(shape=n, rate=k))
def CBexponential_mean(x) :
    n = len(x)
    k = sum(x)
    return(1/gamma(shape=n, rate=k))
def CBexponential_lambda(x) : return(1/CBexponential_mean(x))







## x[i] ~ lognormal(mean, sd), where mean and sd are the mean and standard deviation of the x[i] values
#
## Would like a formula for the mean-stdev parameterization for lognormal, but this awkward strategy doesn't work:
#CBlognormal.mean <- function(x) {
#mu = CBlognormal.mu(x)
#sigma = CBlognormal.sigma(x)
#return(exp(mu %|+|% sigma^2)/2))
#}

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
           
# x[i] ~ F, a continuous but unknown distribution
# N.B. the infinities don't plot, but they are there
def CBnonparametric(x) : return(env(histogram(np.concatenate((x, [-np.inf]))), histogram(np.concatenate((x, [np.inf])))))

# x1[i] ~ normal(mu1, sigma1), x2[j] ~ normal(mu2, sigma2), x1 and x2 are independent
def CBnormal_meandifference(x1, x2) : return(CBnormal_mu(x2) - CBnormal_mu(x1))

# x[i] = Y + error[i],  error[j] ~ F,  F unknown,  Y fixed,  x[i] and error[j] are independent
def CBnonparametric_deconvolution(x, error) : # i.e., the c-box for Y

  def Get_Q( m_in , c_in , k = None) :
    if k is None : k = np.arange((m_in*c_in+1)) 
    def Q_size_GLBL( m ) : return(1 + m + m*(m+1)/2 + m*(m+1)*(m+2)*(3*m+1)/24)
    def Q_size_LoCL( m , c ) : return(1 + c + m*c*(c+1)/2 )
    def Grb_Q( m_in , c_in , Q_list ) : 
      m = max( m_in , c_in )
      c = min( m_in , c_in )
      i_min = Q_size_GLBL( m - 1 ) + Q_size_LoCL( m , c-1 ) + 1
      return(Q_list[i_min:(i_min + m*c)])
  
    def AddingQ( m , Q_list ) :
      Q_list[ Q_size_GLBL( m - 1 ) + 1 ] = 1       
      for c in range(m) :
          i_min = Q_size_GLBL( m - 1 ) + Q_size_LoCL( m , c ) + 1
          Q1 = np.concatenate(( Grb_Q( m-1 , c+1 , Q_list ) , np.repeat(0,(c+1))  ))
          Q2 = np.concatenate(( np.repeat(0,m), Grb_Q( m , c , Q_list )  ))
          Q_list[ i_min:(i_min + m*(c+1)) ] = Q1 + Q2
      return(Q_list[(Q_size_GLBL( m-1 ) + 1):Q_size_GLBL( m )])

    def Bld_Q( m_top ) :
      Q_out = np.repeat(0,Q_size_GLBL( m_top ))
      Q_out[0] = 1
      for m in range(m_top) :
        Q_out[ (Q_size_GLBL( m ) + 1):(Q_size_GLBL( m+1 )) ] = AddingQ( m+1 , Q_out )
      return(Q_out)

    # body of Get_Q
    m = max( m_in , c_in )
    c = min( m_in , c_in )
    return(Grb_Q(m, c, Bld_Q(m))[k+1])

  
  # body of CBnonparametric_deconvolution
  z = []
  for err in error : z = np.append(z, [x - err])
  z.sort()
  Q = Get_Q(len(x), len(error))
  w = Q / sum( Q )
  return(env(mixture(z,w), mixture(np.append(z[1:],[np.inf]),w)))









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

def bernoulli(p) : return(np.random.uniform(size=many) < p)

def beta(a,b) :
    #if (a==0) and (b==0) : return(env(np.repeat(0.0, many), np.repeat(1.0, many)))  # this is [0,1]
    if (a==0) and (b==0) : return(bernoulli(0.5))  # or should it be [0,1]?
    if (a==0) : return(np.repeat(0.0, many))
    if (b==0) : return(np.repeat(1.0, many))            
    return(scipy.stats.beta.rvs(a,b,size=many))

def beta1(m,s) : return(beta(m * (m * (1 - m) / (s**2) - 1), (m * (m * (1 - m) / (s**2) - 1)) * (1/m - 1)))

def betabinomial2(size,v,w) : return(scipy.stats.binom.rvs(size,beta(v,w),size=many))

def betabinomial(size,v,w) : return(scipy.stats.betabinom.rvs(size,v,w,size=many))

def binomial(size,p) : return(scipy.stats.binom.rvs(size,p,size=many))

def chisquared(v) : return(scipy.stats.chi2.rvs(v,size=many))

def delta(a) : return(np.repeat(a,many))

def exponential(rate=1,mean=None) :
    if mean is None : mean = 1/rate
    #rate = 1/mean
    return(scipy.stats.expon.rvs(scale=mean,size=many))

def exponential1(mean=1) :
    return(scipy.stats.expon.rvs(scale=mean,size=many))

def F(df1,df2) : return(scipy.stats.f.rvs(df1,df2,size=many))

def gamma(shape,rate=1,scale=None) :
    if scale is None : scale = 1/rate
    rate = 1/scale
    return(scipy.stats.gamma.rvs(a=shape,scale=1/rate,size=many))

def gammaexponential(shape,rate=1,scale=None) :
    if scale is None : scale = 1/rate
    rate = 1/scale
    #expon(scale=gamma(a=shape, scale=1/rate))
    return(scipy.stats.expon.rvs(scale=1/scipy.stats.gamma.rvs(a=shape,scale=scale,size=many),size=many))

def geometric(m) : return(scipy.stats.geom.rvs(m,size=many))

def gumbel(loc,scale) : return(scipy.stats.gumbel_r.rvs(loc,scale,size=many))

def inversechisquared(v) : return(1/chisquared(v))
    
def inversegamma(shape, scale=None, rate=None) : 
    if scale is None and not rate is None : scale = 1/rate
    return(scipy.stats.invgamma.rvs(a=shape,scale=scale,size=many))

def laplace(a,b) :  return(scipy.stats.laplace.rvs(a,b,size=many))

def logistic(loc,scale) : return(scipy.stats.logistic.rvs(loc,scale,size=many))

def lognormal(m,s) : 
    m2 = m**2; s2 = s**2
    mlog = np.log(m2/np.sqrt(m2+s2))
    slog = np.sqrt(np.log((m2+s2)/m2))
    return(scipy.stats.lognorm.rvs(s=slog,scale=np.exp(mlog),size=many))

def lognormal2(mlog,slog) : return(scipy.stats.lognorm.rvs(s=slog,scale=np.exp(mlog),size=many))

#lognormal = function(mean=NULL, std=NULL, meanlog=NULL, stdlog=NULL, median=NULL, cv=NULL, name='', ...){
#  if (is.null(meanlog) & !is.null(median)) meanlog = log(median)
#  if (is.null(stdlog) & !is.null(cv)) stdlog = sqrt(log(cv^2 + 1))
#  # lognormal(a, b) ~ lognormal2(log(a^2/sqrt(a^2+b^2)),sqrt(log((a^2+b^2)/a^2)))
#  if (is.null(meanlog) & (!is.null(mean)) & (!is.null(std))) meanlog = log(mean^2/sqrt(mean^2+std^2))
#  if (is.null(stdlog) & !is.null(mean) & !is.null(std)) stdlog = sqrt(log((mean^2+std^2)/mean^2))
#  if (!is.null(meanlog) & !is.null(stdlog)) Slognormal0(meanlog,stdlog,name) else stop('not enough information to specify the lognormal distribution')
#  }

def loguniform_solve(m,v) :
  def loguniform_f(a,m,v) : return(a*m*np.exp(2*(v/(m**2)+1)) + np.exp(2*a/m)*(a*m - 2*((m**2) + v)))
  def LUgrid(aa, w) : return(left(aa)+(right(aa)-left(aa))*w/100.0)
  aa = (m - np.sqrt(4*v), m)   # interval
  a = m
  ss = loguniform_f(a,m,v)
  for j in range(4) :
    for i in range(101) :  # 0:100 
      a = LUgrid( aa, i)
      s = abs(loguniform_f(a,m,v))
      if s < ss :
          ss = s
          si = i 
    a = LUgrid(aa, si)
    aa = (LUgrid(aa, si-1), LUgrid(aa, si+1))  # interval
  return(a)

def loguniform(min=None, max=None, minlog=None, maxlog=None, mean=None, std=None) :
    if (min is None) and (not (minlog is None)) : min = np.exp(minlog)
    if (max is None) and (not (maxlog is None)) : max = np.exp(maxlog)  
    if (max is None) and (not (mean is None)) and (not (std is None)) and (not (min is None)) : max = 2*(mean**2 +std**2)/mean - min
    if (min is None) and (max is None) and (not (mean is None)) and (not(std is None)) :
        min = loguniform_solve(mean,std**2)
        max = 2*(mean**2 +std**2)/mean - min
    return(scipy.stats.loguniform.rvs(min, max, size=many))

def loguniform1(m,s) : return(loguniform(mean=m, std=s))

def negativebinomial(size,prob) : return(scipy.stats.nbinom.rvs(size,prob,size=many))

def normal(m,s) : return(scipy.stats.norm.rvs(m,s, size=many))

def pareto(mode, c) : return(scipy.stats.pareto.rvs(c,scale=mode,size=many))

def poisson(m) : return(scipy.stats.poisson.rvs(m,size=many))

def powerfunction(b,c) : return(scipy.stats.powerlaw.rvs(c,scale=b,size=many))

# parameterisation of rayleigh differs from that in pba.r
def rayleigh(loc,scale) : return(scipy.stats.rayleigh.rvs(loc,scale,size=many))

def sawinconrad(min, mu, max) : # WHAT are the 'implicit constraints' doing?     
  def sawinconradalpha01(mu) :
      def f(alpha) : return(1/(1-1/np.exp(alpha)) - 1/alpha - mu)
      if np.abs(mu-0.5)<0.000001 : return(0)      
      return(uniroot(f,np.array((-500,500))))
  def qsawinconrad(p, min, mu, max) : 
        alpha = sawinconradalpha01((mu-min)/(max-min))
        if np.abs(alpha)<0.000001 : return(min+(max-min)*p) 
        else : min+(max-min)*((np.log(1+p*(np.exp(alpha)-1)))/alpha)
  a = left(min);   b = right(max)
  c = left(mu);    d = right(mu)
  if c<a : c = a   # implicit constraints
  if b<d : d = b
  #return(qsawinconrad(np.random.uniform(size=many), min, mu, max))
  return(qsawinconrad(np.random.uniform(size=many), min, mu, max))
  
def student(v) : return(scipy.stats.t.rvs(v,size=many))

def uniform(a,b) : return(scipy.stats.uniform.rvs(a,b-a,size=many)) # who parameterizes like this?!?!

def triangular(min,mode,max) : return(np.random.triangular(min,mode,max,size=many)) # cheating: uses random rather than scipy.stats

def histogram(x) : return(x[(np.trunc(scipy.stats.uniform.rvs(size=many)*len(x))).astype(int)])

def mixture(x,w=None) :
    if w is None : w = np.repeat(1,len(x))
    print(many)
    r = np.sort(scipy.stats.uniform.rvs(size=many))[::-1]
    x = np.concatenate(([x[0]],x))
    w = np.cumsum(np.concatenate(([0],w)))/np.sum(w)
    u = []
    j = len(x)-1
    for p in r : 
        while True :
            if w[j] <= p : break
            j = j - 1
        u = np.concatenate(([x[j+1]],u))
    return(u[np.argsort(scipy.stats.uniform.rvs(size=len(u)))])






def testing():

  ###############################################################################
  # Testing and exercising the functions
  ###############################################################################

  ###############################################################################
  # Maximum likelihood estimation for lognormal using scipy.stats.fit()

  # So far, I'm not impressed with scipy.stats ML fitting. What am I doing wrong?
  # It looks like I'm not doing anything wrong.  It just doesn't work so well 
  # with really small data sets.

  print('****** 1')

  # N = 10
  w = np.array([2.912,2.5565,2.9077,4.6462,3.5,2.2677,4.6362,3.017,3.9792,4.6102])
  print(scipy.stats.lognorm.fit(w)) # (10.6565, 2.26770, 0.03194)
  L = scipy.stats.lognorm.rvs(s=10.656, loc=2.2677, scale=0.0319,size=many)
  LL = MLlognormal(w)
  LLL = sMLlognormal(w)
  pl((-100,1e3)); edf(L,'r'); edf(LL); edf(LLL,'g'); #edf(w)
  edf(LLL,'g'); edf(w)
  print(np.mean(L));print(np.mean(LL));print(np.mean(LLL));print(np.mean(w))
  plt.show()

  # N = 30
  W = scipy.stats.lognorm.rvs(s=2,scale=np.exp(3),size=30)
  print(scipy.stats.lognorm.fit(W)) 
  L = scipy.stats.lognorm.rvs(*scipy.stats.lognorm.fit(W),size=many)
  LL = MLlognormal(W)
  LLL = sMLlognormal(W)
  pl((-100,1e3)); edf(L,'r'); edf(LL); edf(LLL,'g'); edf(W,'k')
  #edf(LLL,'g'); edf(W)
  print(np.mean(L));print(np.mean(LL));print(np.mean(LLL));print(np.mean(W))
  plt.show()

  # N = 100
  W = scipy.stats.lognorm.rvs(s=2,scale=np.exp(3),size=100)
  print(scipy.stats.lognorm.fit(W)) 
  L = scipy.stats.lognorm.rvs(*scipy.stats.lognorm.fit(W),size=many)
  LL = MLlognormal(W)
  LLL = sMLlognormal(W)
  pl((-100,1e3)); edf(L,'r'); edf(LL); edf(LLL,'g'); edf(W,'k')
  #edf(LLL,'g'); edf(W)
  print(np.mean(L));print(np.mean(LL));print(np.mean(LLL));print(np.mean(W))
  plt.show()

  ###############################################################################
  # Miscellaneous PERT and Fermi estimates

  print('****** 2')

  x = np.random.normal(size=25)
  edf(antweiler(x))
  
  edf(betapert(5, 10, 6),'r')

  ferminorm(12, 16)  # array([14., 1.560])
  edf(normal(*ferminorm(12, 16)),'g') 

  fermilnorm(16, 32)  # array([3.12, 0.27])
  edf(lognormal2(*fermilnorm(16, 32)),'k') 

  plt.show()

  ###############################################################################
  # Fermi and KS confidence bands

  print('****** 3')

  bOt = 0.001
  tOp = 0.999
  m,s = ferminorm(2,10,100,.9)
  BOT = scipy.stats.norm.ppf(bOt,m,s)
  TOP = scipy.stats.norm.ppf(tOp,m,s)
  n=normal(m,s)
  f=ferminormconfband(2,10,100)
  red(n); edf(f)

  m,s = ferminorm(12,20,100,.9)
  n=normal(m,s)
  f=ferminormconfband(12,20,100,bOt=0.00001,tOp=0.99999)
  red(n); edf(f)

  mlog,slog = fermilnorm(22,27,50,.9)
  n = lognormal2(mlog,slog)
  f = fermilnormconfband(22,27,50)  # n = 50, lognormal
  red(n); edf(f)

  w = 36 + 2*np.random.normal(size=25)
  k = ks(w)
  edf(k); red(w)

  plt.show()

  ###############################################################################
  # beta distribution constructors should be able to handle funky parameters

  print('****** 4')

  x = beta(1,1)   # U(0,1)
  print(x.mean()) # 0.5

  x = beta(1,0)   # 1
  print(x.mean()) # 1

  x = beta(0,1)   # 0
  print(x.mean()) # 0

  x = beta(0,0)   # bernoulli(0.5), or should it be [0,1]?   
  print(x.mean()) # 0.5, or the interval [0,1]

  ###############################################################################
  # Pareto

  print('****** 5')

  # picture in Wikipedia https://en.wikipedia.org/wiki/Pareto_distribution
  m=1;pl((0,5)); edf(pareto(m,1));edf(pareto(m,2));edf(pareto(m,3));edf(pareto(m,3000));plt.title('pareto');plt.show() 

  m=2;pl((0,5)); edf(pareto(m,1));edf(pareto(m,2));edf(pareto(m,3));edf(pareto(m,3000));plt.title('pareto');plt.show()

  ###############################################################################
  # power function distribution

  print('****** 6')

  pl((0,1)); pl((0,1)); pl((0,1))

  edf(powerfunction(1,1)); plt.show()
  edf(powerfunction(2,1)); plt.show()
  edf(powerfunction(3,1)); plt.show()
  edf(powerfunction(4,1)); plt.show()
      
  edf(powerfunction(1,2)); plt.show()
  edf(powerfunction(2,2)); plt.show()
  edf(powerfunction(3,2)); plt.show()
  edf(powerfunction(4,2)); plt.show()

  edf(powerfunction(1,3)); plt.show()
  edf(powerfunction(2,3)); plt.show()
  edf(powerfunction(3,3)); plt.show()
  edf(powerfunction(4,3)); plt.show()

  ###############################################################################
  # Laplace

  print('****** 7')

  # picture on Wikipedia https://en.wikipedia.org/wiki/Laplace_distribution
  m=0;pl((-10,10));edf(laplace(m,1));edf(laplace(m,2));edf(laplace(m,4));edf(laplace(-5,4));plt.title('laplace');plt.show()  
  
  ###############################################################################
  # I don't think the fits to loguniform are correct

  print('****** 8')

  w = np.random.uniform(2,5,size=200); edf(w); edf(MMloguniform(w)); edf(MLloguniform(w)); plt.show()
  w = np.random.uniform(2,5,size=20); edf(w); edf(MMloguniform(w)); edf(MLloguniform(w)); plt.show()

  ###############################################################################
  # are we sure the asterisk operator works?  (it's the best thing about Python)
  # the red and green fitted distributions should be the same, modulo MC error

  print('****** 9')

  def AMLgumbel(x) :
      loc, scale = scipy.stats.gumbel_r.fit(x)
      return(gumbel(loc,scale))

  def BMLgumbel(x) : return(gumbel(*scipy.stats.gumbel_r.fit(w)))    

  w = np.array([2.91247063, 2.55651104, 2.90768457, 4.64622234, 3.49995966,
        2.26770086, 4.63619271, 3.01703563, 3.97919485, 4.61017778,
        2.00292333, 3.13348299, 4.68998771, 2.30031397, 2.14102056,
        4.23825192, 2.56982047, 4.86396995, 3.79969706, 4.00203139])
  g1 = AMLgumbel(w); g2 = BMLgumbel(w)
  edf(w); edf(g1,'r'); edf(g2,'k'); plt.show()

  w = np.array([2.91247063, 2.55651104, 2.90768457, 4.64622234, 3.49995966,
        2.26770086, 4.63619271, 3.01703563, 3.97919485, 4.61017778])
  g1 = AMLgumbel(w); g2 = BMLgumbel(w)
  edf(w); edf(g1,'r'); edf(g2,'k'); plt.show()

  G1,G2 = scipy.stats.gumbel_r.fit(w)
  print(G1,G2)
  print(*scipy.stats.gumbel_r.fit(w))
  print(scipy.stats.gumbel_r.fit(w))

  ###############################################################################
  # parameterisations: Rayleigh, power function, gamma, inverse gamma, gammaexponential 

  print('****** 10')

  # parameterization of Rayliegh distributions does NOT match Risk Calc and pba.r

  # raleigh() ***************************************


  # parameterization for power function distributions seems to match Risk Calc and pba.r

  edf(powerfunction(1,1));edf(powerfunction(2,1));edf(powerfunction(3,1));edf(powerfunction(4,1))
  edf(powerfunction(1,2));edf(powerfunction(2,2));edf(powerfunction(3,2));edf(powerfunction(4,2))
  edf(powerfunction(1,3));edf(powerfunction(2,3));edf(powerfunction(3,3));edf(powerfunction(4,3));plt.show()

  # parameterizations for gamma and inversegamma distributions don't match Risk Calc
  # but I've just updated pba.r [15 Nov 2024] so it now agrees with these conventions

  # The parameterizations for gamma and inversegamma distributions 
  # match with their Wikipedia articles (as of 15 November 2024).
  # Note that the wrinkle is that one's scale is the other's rate.

  ######### examples from the Wikipedia pages

  print('display Reyleigh examples')
  # https://en.wikipedia.org/wiki/Rayleigh_distribution
  #blue(rayleigh(0.5)); green(rayleigh(1)); red(rayleigh(2)); cyan(rayleigh(3)); purple(rayleigh(4)); plt.show()

  # https://en.wikipedia.org/wiki/Gamma_distribution
  red(gamma(shape=1, scale=2));orange(gamma(shape=2, scale=2));yellow(gamma(shape=3, scale=2));green(gamma(shape=5, scale=1)); black(gamma(shape=9, scale=0.5));blue(gamma(shape=7.5, scale=1));purple(gamma(shape=0.5, scale=1));plt.show()

  # https://en.wikipedia.org/wiki/Inverse-gamma_distribution
  red(inversegamma(shape=1, scale=1));green(inversegamma(shape=2, scale=1));blue(inversegamma(shape=3, scale=1));cyan(inversegamma(shape=3, scale=0.5)); plt.show()

  # there is no Wikipedia article on gammaexponential, so instead we can check the
  # picture created below with the picture made by R (immediately further down)
  #
  # Python
  many = 10000
  g = gammaexponential(shape=1, scale=1); pl(0,20); blue(g)
  g = gammaexponential(shape=2, scale=1); pl(0,20); red(g)
  g = gammaexponential(shape=1, scale=2); pl(0,20); black(g)
  g = gammaexponential(shape=2, scale=2); pl(0,20); yellow(g)
  g = gammaexponential(shape=1, scale=11); pl(0,20); cyan(g)
  g = gammaexponential(shape=1, scale=0.1); pl(0,20); purple(g)

  # # R
  # source('pba BETTER.r')
  # rbyc()
  # pl(0,20)
  # g = gammaexponential(shape=1, scale=1);   blue(g)
  # g = gammaexponential(shape=2, scale=1);   red(g)
  # g = gammaexponential(shape=1, scale=2);   black(g)
  # g = gammaexponential(shape=2, scale=2);   yellow(g)
  # g = gammaexponential(shape=1, scale=11);  cyan(g)
  # g = gammaexponential(shape=1, scale=0.1); purple(g)

  ######### moments are correct now
  shape = 4
  scale = 6
  rate = 1/scale
  #rbyc(3,2)
  tOp = 0.99999

  print('display Reyleigh examples')
  #r = rayleigh(scale)
  #print(scale * np.sqrt(np.pi/2),   r.mean())           # mean
  #print((4-np.pi)*scale**2/2,       r.var())            # var 

  p = powerfunction(scale,shape)  # Risk Calc is the reference for the moments [scale=b,shape=c]
  print(scale/(1+1/shape),   p.mean())                  # mean
  print(scale**2/((1+2/shape)*(shape+1)**2), p.var())   # var # maybe not defined, unless truncated

  g = gamma(shape=shape, scale=scale)
  print(shape * scale,      g.mean())                   # mean
  print(shape * scale**2,   g.var())                    # var 

  ig = inversegamma(shape=shape, scale=scale)
  print(scale / (shape - 1),               ig.mean())   # mean
  print(scale**2 / ((shape-1)**2*(shape-2)), ig.var())    # var

  ######### reciprocation

  def compare(x,y) : edf(x,'b',lw=7); edf(y,'y',lw=3); plt.show()

  SCALE = 6
  SHAPE = 4
  RATE = 1/SCALE

  compare(1/zbuff(gamma(shape=SHAPE, rate=RATE)), inversegamma(shape=SHAPE, rate=1/RATE))
  compare(1/zbuff(gamma(shape=SHAPE, rate=RATE)), inversegamma(shape=SHAPE, scale=1/SCALE))
  compare(1/zbuff(gamma(shape=SHAPE, rate=RATE)), inversegamma(shape=SHAPE, rate=SCALE))
  compare(1/zbuff(gamma(shape=SHAPE, rate=RATE)), inversegamma(shape=SHAPE, scale=RATE))
  compare(1/zbuff(gamma(shape=SHAPE, scale=SCALE)), inversegamma(shape=SHAPE, scale=1/SCALE))
  compare(1/zbuff(gamma(shape=SHAPE, scale=SCALE)), inversegamma(shape=SHAPE, rate=1/RATE))
  compare(1/zbuff(gamma(shape=SHAPE, scale=SCALE)), inversegamma(shape=SHAPE, scale=RATE))
  compare(1/zbuff(gamma(shape=SHAPE, scale=SCALE)), inversegamma(shape=SHAPE, rate=SCALE))

  # Using the Monte Carlo model for distributions simplifies things a little here.
  # You can just reciprocate a gamma and you'll get the appropriate inverse gamma.
  # You don't even need the zbuff function to protect against division by zero, as 
  # the Monte Carlo deviates will never be exactly zero.  Having an inversegamma 
  # constructor is handy in Risk Calc and pba.r because their distribution models 
  # are discretizations rather than Monte Carlo assemblages.

  #def gamma(shape,rate=1,scale=None) :
  #    if scale is None : scale = 1/rate
  #    rate = 1/scale
  #    return(scipy.stats.gamma.rvs(a=shape,scale=1/rate,size=many))

  def inversegammaRECIP(shape, scale=None, rate=None) : 
      if scale is None : scale = 1/rate
      return(1/gamma(shape=shape,scale=1/scale))

  def inversegammaSCIPY(shape, scale=None, rate=None) : 
      if scale is None and not rate is None : scale = 1/rate
      return(scipy.stats.invgamma.rvs(a=shape,scale=scale,size=many))

  a = inversegammaRECIP(scale=SCALE, shape=SHAPE)
  b = inversegammaSCIPY(scale=SCALE, shape=SHAPE)
  compare(a,b)

  ###############################################################################
  # c-box and distribution-free p-box constructors

  print('****** 11')

  def sh(x,t,Data=None) : 
      if Data is None : Data = data
      plt.title(t)
      edf(Data,'y')
      edf(x)
      plt.show()

  k = 22
  m = 11
  n = k + m
  fdata = np.concatenate((m*[0],k*[1]))
  bdata = np.random.uniform(size=25) > 0.35
  idata = np.round(np.random.uniform(size=25) * 16)
  data = np.random.uniform(size=25) * 30
  x2 = 5 + np.random.uniform(size=25) * 30
  error = np.random.normal(size=25)

  x=km(k,m);                                    sh(x,'km',fdata)
  x=KN(k,n);                                    sh(x,'KN',fdata)
  x=FKN(k,n);                                   sh(x,'FKN',fdata)
  x=CBbernoulli(bdata);                         sh(x,'CBbernoulli',bdata)
  x=CBbernoulli_p(bdata);                       sh(x,'CB p',bdata)             
  x=CBbinomial(n,idata);                        sh(x,'CBbinomial(n)',idata)
  x=CBbinomial_p(n,idata);                      sh(x,'CB p',idata)
  #x=CBbinomialnp(x);                           sh(x,'CB')
  #x=CBbinomialnp_n(x);                         sh(x,'CB')
  #x=CBbinomialnp_p(x);                         sh(x,'CB') 
  x=CBpoisson(idata);                           sh(x,'CBpoisson',idata)
  x=CBpoisson_mean(idata);                      sh(x,'CB mean',idata)
  x=CBexponential(data);                        sh(x,'CBexponential')          
  x=CBexponential_mean(data);                   sh(x,'CB mean')                
  x=CBexponential_lambda(data);                 sh(x,'CB lambda')              
  x=CBnormal(data);                             sh(x,'CBnormal')
  x=CBnormal_mu(data);                          sh(x,'CB mu')
  x=CBnormal_sigma(data);                       sh(x,'CB sigma')
  x=CBlognormal(data);                          sh(x,'CBlognormal')
  x=CBlognormal_mu(data);                       sh(x,'CB mu')
  x=CBlognormal_sigma(data);                    sh(x,'CB sigma')
  x=CBuniform(data);                            sh(x,'CBuniform')
  x=CBuniform_midpoint(data);                   sh(x,'CB midpoint')
  x=CBuniform_width(data);                      sh(x,'CB width')
  x=CBuniform_minimum(data);                    sh(x,'CB minimum')
  x=CBuniform_maximum(data);                    sh(x,'CB maximum')
  x=CBnonparametric(data);                      sh(x,'CBnonparametric')
  x=CBnormal_meandifference(data, x2);          sh(x,'CB normal mean difference, should be ~5')
  x=CBnonparametric_deconvolution(data, error); sh(x,'CB deconvolution')       ##

  ###############################################################################
  # maxent constructors

  print('****** 12')

  def sh(x,t) : 
      edf(x)
      plt.title(t)
      plt.show()

  x=MEminmax(min=10, max=14);              sh(x,'MEminmax(10,14)')
  x=MEminmaxmean(min=10, max=14, mean=11); sh(x,'MEminmaxmean(10,14,11)')      ##
  x=MEmeansd(mean=20, sd=1);               sh(x,'MEmeansd(20,1)')
  x=MEminmean(min=1,mean=2);               sh(x,'MEminmean(1,2)')
  #x=MEmeangeomean(mean=12, geomean=10);    sh(x,'MEmeangeomean(12,10)')       ##
  x=MEdiscretemean(x=[1,2,3,4,5,6],mu=2.3);sh(x,'discretemean([1,2,3,4,5,6],2.3)') # e.g., MEdiscretemean(1:10,2.3)
  x=MEquantiles(v=np.array((0,1,2,3,5)),p=np.array((0,.03,.3,.36,1))); sh(x,'MEquantiles([0,1,2,3,5],p=[0,.03,.3,.36,1]')
  x=MEdiscreteminmax(min=21,max=45);       sh(x,'MEdiscreteminmax(21,45)')
  x=MEmeanvar(10,3);                       sh(x,'MEmeanvar(10,3)')
  x=MEminmaxmeansd(10,20,13,1);            sh(x,'MEminmaxmeansd(10,20,13,1)')
  x=MEmmms(min=10, max=20, mean=13, sd=2); sh(x,'MEmmms(10,20,13,2)')
  x=MEminmaxmeanvar(0,1,0.8,0.1);          sh(x,'MEminmaxmeanvar(0,1,0.8,0.1)')


  ###############################################################################
  # method of matching moments (MoM) constructors

  print('****** 13')

  def sh(x,t,Data=None) : 
      if Data is None : Data = data
      plt.title(t)
      edf(Data,'y')
      edf(x)
      plt.show()

  data = np.random.uniform(size=25)
  datat = 2 * np.random.normal(size=25)
  data2 = 2 * data
  data10 = np.round(data*10) 
  datap = data + 1
  data10p = np.round(data*10) + 1
  data100 = 1+np.round(data * 100)
  dataBB = np.array((0,0,0,0,0,0,0,0,0,0,0,0,3,24,104,286,670,1033,1343,1112,829,478,181,45,7))
  N = 12

  x=MMbernoulli(data);                sh(x,'MMbernoulli(data)')
  x=MMbeta(data);                     sh(x,'MMbeta(data)')
  x=MMbetabinomial(N,dataBB);         sh(x,'MMbetabinomial(12,dataBB)',dataBB)  
  x=MMbinomial(data10p);              sh(x,'MMbinomial(data100)',data10) 
  x=MMchisquared(data);               sh(x,'MMchisquared(data)')
  x=MMexponential(data);              sh(x,'MMexponential(data)')
  x=MMF(data2);                       sh(x,'MMF(data2)',data2)               
  x=MMgamma(data10);                  sh(x,'MMgamma(data10)',data10)
  x=MMgaussian(data);                 sh(x,'MMgaussian(data)')
  x=MMgeometric(data10p);             sh(x,'MMgeometric(data10p)',data10p)
  x=MMpascal(data10p);                sh(x,'MMpascal(data10p)',data10p)
  x=MMgumbel(data);                   sh(x,'MMgumbel(data)')
  x=MMextremevalue(data);             sh(x,'MMextremevalue(data)')
  x=MMlognormal(data);                sh(x,'MMlognormal(data)')
  x=MMlaplace(data);                  sh(x,'MMlaplace(data)')
  x=MMdoubleexponential(data);        sh(x,'MMdoubleexponential(data)')
  x=MMlogistic(data);                 sh(x,'MMlogistic(data)')
  x=MMloguniform(data);               sh(x,'MMloguniform(data)')
  x=MMnormal(data);                   sh(x,'MMlognormal(data)')
  x=MMpareto(data);                   sh(x,'MMpareto(data)')
  x=MMpoisson(data);                  sh(x,'MMpoisson(data)')
  x=MMpowerfunction(data);            sh(x,'MMpowerfunction(data)')
  x=MMt(datat);                       sh(x,'MMt(datat)',datat)                        
  x=MMstudent(datat);                 sh(x,'MMstudent(datat)',datat)                  
  x=MMuniform(data);                  sh(x,'MMuniform(data)')
  x=MMrectangular(data);              sh(x,'MMrectangular(data)')
  x=MMtriangular(data);               sh(x,'MMtriangular(data)')

  ###############################################################################
  # maximum likelihood constructors

  print('****** 14')




  ###############################################################################
  # alternative maximum likelihood constructors

  print('****** 15')


  ###############################################################################
  # Bayes constructors

  print('****** 16')


  ###############################################################################
  # maximum a posteriori constructors

  print('****** 17')

















  ###############################################################################
  # bestiary of precise distributions

  print('****** 18')

  def sh(x,t) : 
      edf(x)
      plt.title(t)
      plt.show()

  x=bernoulli(p=0.25);               sh(x,'bernoulli(p=0.25)') 
  x=beta(a=2,b=3) ;                  sh(x,'beta(a=2,b=3)') 
  x=betabinomial2(size=10,v=2,w=3);  sh(x,'betabinomial2(size=10,v=2,w=3)') 
  x=betabinomial(size=10,v=2,w=3);   sh(x,'betabinomial(size=10,v=2,w=3)')  
  x=binomial(12,0.4);                sh(x,'binomial(size=12,p=0.4)')
  x=chisquared(v=6);                 sh(x,'chisquared(v=6)')
  x=exponential(mean=2);             sh(x,'exponential(mean=2)') 
  x=F(6,11);                         sh(x,'F(df1=6,df2=11)')
  x=gamma(shape=4,rate=2);           sh(x,'gamma(shape=4,rate=2)')
  x=gammaexponential(shape=4,rate=2);sh(x,'gammaexponential(shape=4,rate=2)')
  x=geometric(m=0.3);                sh(x,'geometric(m=0.3)')
  x=gumbel(2,4);                     sh(x,'gumbel(loc=2,scale=4)')
  x=inversechisquared(14);           sh(x,'inversechisquared(df=14)') 
  x=inversegamma(shape=2,scale=4);   sh(x,'inversegamma(shape=2,scale=4)')
  x=laplace(a=4,b=5);                sh(x,'laplace(a=4,b=5)') 
  x=logistic(2,3);                   sh(x,'logistic(loc=2,scale=3)')
  x=lognormal(m=2,s=1);              sh(x,'lognormal(m=10,s=1)')
  x=lognormal2(mlog=-2,slog=1);      sh(x,'lognormal2(mlog=-2,slog=1)')
  x=loguniform(min=2, max=6);        sh(x,'loguniform(min=2, max=6)')
  x=negativebinomial(size=10,prob=0.25); sh(x,'negativebinomial(size=10,prob=0.25)') 
  x=normal(m=5,s=1) ;                sh(x,'normal(m=5,s=1)')
  x=pareto(mode=3, c=2);             sh(x,'pareto(mode=3, c=2)')
  x=poisson(m=4);                    sh(x,'poisson(m=4)')
  #x=rayleigh(4,3);                   sh(x,'rayleigh(4,3)')
  x=sawinconrad(2,4,9) ;             sh(x,'student(2,4,9)')
  x=student(v=5) ;                   sh(x,'student(v=5)')
  x=triangular(2,5,11);              sh(x,'triangular(2,5,11)')
  x=uniform(a=2,b=4) ;               sh(x,'uniform(a=2,b=4)') 

  print('****** 19')

  x = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9])
  w = np.array([ 1,  5,  1,  1,  1,  1,  2,  3, 15])
  q = mixture(x,w)
  h = histogram(x)
  edf(q)
  edf(h)
  plt.show()
  print(np.mean(x),np.mean(h), 'but', np.mean(q), np.sum(x * (w/np.sum(w))))

  print('****** 20')

  
  """


  ###############################################################################
  # Repeat some of the above calculations in R to compare and check the results
  ###############################################################################

  ###############################################################################
  # power function distribution

  rbyc(3,4)
  powerfunction(1,1);powerfunction(2,1);powerfunction(3,1);powerfunction(4,1)
  powerfunction(1,2);powerfunction(2,2);powerfunction(3,2);powerfunction(4,2)
  powerfunction(1,3);powerfunction(2,3);powerfunction(3,3);powerfunction(4,3)

  ###############################################################################
  # gamma-exponential compound distribution

  #source('pba BETTER.r')
  rbyc()
  pl(0,20)
  g = gammaexponential(shape=1, scale=1);   blue(g)
  g = gammaexponential(shape=2, scale=1);   red(g)
  g = gammaexponential(shape=1, scale=2);   black(g)
  g = gammaexponential(shape=2, scale=2);   yellow(g)
  g = gammaexponential(shape=1, scale=11);  cyan(g)
  g = gammaexponential(shape=1, scale=0.1); purple(g)

  ###############################################################################
  # maxent

  cat('****** 12\n')

  sh = function(x,t) { pl(x); edf(x,1); title(t) }
  rbyc(3,4)
  x=MEminmax(min=10, max=14);              sh(x,'MEminmax(10,14)')
  x=MEminmaxmean(min=10, max=14, mean=11); sh(x,'MEminmaxmean(10,14,11)')      ##
  x=MEmeansd(mean=20, sd=1);               sh(x,'MEmeansd(20,1)')
  x=MEminmean(min=1,mean=2);               sh(x,'MEminmean(1,2)')
  #x=MEmeangeomean(mean=12, geomean=10);    sh(x,'MEmeangeomean(12,10)')       ##
  x=MEdiscretemean(x=c(1,2,3,4,5,6),mu=2.3);sh(x,'discretemean(c(1,2,3,4,5,6),2.3)') # e.g., MEdiscretemean(1:10,2.3)
  x=MEquantiles(c(0,1,2,3,5),c(0,.03,.3,.36,1)); sh(x,'MEquantiles([0,1,2,3,5],p=[0,.03,.3,.36,1]')
  x=MEdiscreteminmax(min=21,max=45);       sh(x,'MEdiscreteminmax(21,45)')
  x=MEmeanvar(10,3);                       sh(x,'MEmeanvar(10,3)')
  x=MEminmaxmeansd(10,20,13,1);            sh(x,'MEminmaxmeansd(10,20,13,1)')
  x=MEmmms(min=10, max=20, mean=13, sd=2); sh(x,'MEmmms(10,20,13,2)')
  x=MEminmaxmeanvar(0,1,0.8,0.1);          sh(x,'MEminmaxmeanvar(0,1,0.8,0.1)')

  ###############################################################################
  # method of matching moments

  cat('****** 13\n')

  sh = function(x,t,Data=NULL) { 
    if (is.null(Data)) Data = data
    pl(x)
    title(t)
    edf(Data,col='green')
    edf(x)
    }

  data = runif(25)
  datat = 2 * rnorm(25)
  data2 = 2 * data
  data10 = round(data*10)
  datap = data + 1
  data10p = round(data*10) + 1
  data100 = 1+round(data * 100)
  N = int(max(data100))

  rbyc(5,5)
  x=MMbernoulli(data);                sh(x,'MMbernoulli(data)')
  x=MMbeta(data);                     sh(x,'MMbeta(data)')
  x=MMbetabinomial(N,data100);        sh(x,'MMbetabinomial(int(max(data)),data100)',data100)  
  x=MMbinomial(data10p);              sh(x,'MMbinomial(data100)',data10) 
  x=MMchisquared(data);               sh(x,'MMchisquared(data)')
  x=MMexponential(data);              sh(x,'MMexponential(data)')
  x=MMF(data2);                       sh(x,'MMF(data2)',data2)               
  x=MMgamma(data10);                  sh(x,'MMgamma(data10)',data10)
  x=MMgaussian(data);                 sh(x,'MMgaussian(data)')
  x=MMgeometric(data10p);             sh(x,'MMgeometric(data10p)',data10p)
  x=MMpascal(data10p);                sh(x,'MMpascal(data10p)',data10p)
  x=MMgumbel(data);                   sh(x,'MMgumbel(data)')
  x=MMextremevalue(data);             sh(x,'MMextremevalue(data)')
  x=MMlognormal(data);                sh(x,'MMlognormal(data)')
  x=MMlaplace(data);                  sh(x,'MMlaplace(data)')
  x=MMdoubleexponential(data);        sh(x,'MMdoubleexponential(data)')
  x=MMlogistic(data);                 sh(x,'MMlogistic(data)')
  x=MMloguniform(data);               sh(x,'MMloguniform(data)')
  x=MMnormal(data);                   sh(x,'MMlognormal(data)')
  x=MMpareto(data);                   sh(x,'MMpareto(data)')
  x=MMpoisson(data);                  sh(x,'MMpoisson(data)')
  x=MMpowerfunction(data);            sh(x,'MMpowerfunction(data)')
  #x=MMt(datat);                       sh(x,'MMt(datat)',datat)                        
  x=MMstudent(datat);                 sh(x,'MMstudent(datat)',datat)                  
  x=MMuniform(data);                  sh(x,'MMuniform(data)')
  #x=MMrectangular(data);              sh(x,'MMrectangular(data)')
  x=MMtriangular(data);               sh(x,'MMtriangular(data)')

  ###############################################################################
  # bestiary

  sh = function(x,t) {
      edf(x,new=TRUE)
      title(t)
      }
  rbyc(5,5)
  x=bernoulli(p=0.25);               sh(x,'bernoulli(p=0.25)') 
  x=beta(2,3) ;                      sh(x,'beta(2,3)') 
  x=betabinomial(n=10,2,3);          sh(x,'betabinomial(size=10,2,3)')  
  x=binomial(12,0.4);                sh(x,'binomial(size=12,0.4)')
  x=chisquared(6);                   sh(x,'chisquared(6)')
  x=exponential(mean=2);             sh(x,'exponential(mean=2)') 
  x=F(6,11);                         sh(x,'F(df1=6,df2=11)')
  x=gamma(shape=4,rate=2);           sh(x,'gamma(shape=4,rate=2)')
  x=gammaexponential(shape=4,rate=2);sh(x,'gammaexponential(shape=4,rate=2)')
  x=geometric(prob=0.3);             sh(x,'geometric(prob=0.3)')
  x=gumbel(2,4);                     sh(x,'gumbel(loc=2,scale=4)')
  x=laplace(a=4,b=5);                sh(x,'laplace(a=4,b=5)') 
  x=logistic(2,3);                   sh(x,'logistic(loc=2,scale=3)')
  x=lognormal(m=2,s=1);              sh(x,'lognormal(m=10,s=1)')
  x=SL(meanlog=-2,stdlog=1);         sh(x,'SL(meanlog=-2,stdlog=1)')  # lognormal should work here instead of SL
  x=loguniform(min=2, max=6);        sh(x,'loguniform(min=2, max=6)')
  x=negativebinomial(size=10,prob=0.25); sh(x,'negativebinomial(size=10,prob=0.25)') 
  x=normal(m=5,s=1) ;                sh(x,'normal(m=5,s=1)')
  x=pareto(mode=3, c=2);             sh(x,'pareto(mode=3, c=2)')
  x=poisson(4);                      sh(x,'poisson(4)')
  x=powerfunction(4,3);              sh(x,'powerfunction(4,3)')
  x=rayleigh(4,3);                   sh(x,'rayleigh(4,3)')
  x=sawinconrad(2,4,9) ;             sh(x,'student(2,4,9)')
  x=student(df=5) ;                  sh(x,'student(df=5)')
  x=triangular(2,5,11);              sh(x,'triangular(2,5,11)')
  x=uniform(2,4) ;                   sh(x,'uniform(2,4)') 


  """
