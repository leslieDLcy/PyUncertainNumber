from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numbers import Number
import scipy.stats as sps
import pyuncertainnumber.pba.intervals.intervalOperators as io
from ..pba.intervals import Interval
from pyuncertainnumber import pba
from ..pba.distributions import Distribution as D
from ..pba.distributions import named_dists
from ..pba.pbox_parametric import named_pbox
import functools
from .core import makeUN

if TYPE_CHECKING:
    from pyuncertainnumber import Interval
    from pyuncertainnumber import UncertainNumber

""" statistical inference functions from data for the UncertainNumber class. """


@makeUN
def fit(method: str, family: str, data: np.ndarray) -> UncertainNumber:
    """parametric estimator to fit a distribution from data

    args:
        method (str): method of fitting, e.g., {'mle' or 'mom'} 'entropy', 'pert', 'fermi', 'bayesian'

        family (str): distribution family to be fitted

        data (np.ndarray): data to be fitted


    return:
        - UncertainNumber object

    example:
        >>> # precise data
        >>> pun.fit('mle', 'norm', np.random.normal(0, 1, 100))
        >>>  # imprecise data
        >>> precise_sample = sps.expon(scale=1/0.4).rvs(15)
        >>> imprecise_data = pba.I(lo = precise_sample - 1.4, hi=precise_sample + 1.4)
        >>> pun.fit('mom', family='exponential', data=imprecise_data)


    .. seealso::
        :func:`pyuncertainnumber.pba.KS_bounds` : a non-parametric charactearisation method using Kolmogorov-Smirnov bounds
    """
    match method:
        case "mle":
            try:
                return named_mle_methods.get(family)(data)
            except:
                return named_smle_methods.get(family)(data)
        case "mom":
            return named_mom_methods.get(family)(data)
        case _:
            raise ValueError("method not supported")


def makedist(shape: str):
    """a decorator to transform a `sps.dist` into `Distribution` objects"""

    def decorator_make_dist(func):
        @functools.wraps(func)
        def wrapper_decorator(*args, **kwargs):  # input array x
            return D.dist_from_sps(func(*args, **kwargs), shape)

        return wrapper_decorator

    return decorator_make_dist


###############################################################################
# Method-of-Moment distribution constructors (matching central moments of x)
###############################################################################
""" hint: given interval data x, return pbox """


def MMbernoulli(x):
    """a first attempt to Maximum likelihood estimation for Bernoulli distribution which accepts both precise and imprecise data;

    #! the example of `singleparam` pattern
    #! to change, add the 'interval_measurement' decorator
    note:
        the attempt is successful per se, but not accommodating to the top-level calling signature yet.

        - precise data returns precise distrubution
        - imprecise data need to be in Interval type to return a pbox
        - interval data can return either a precise distribution or a pbox
    """
    if isinstance(x, sps.CensoredData | np.ndarray | list):
        return D.dist_from_sps(sps.expon(*sps.bernoulli.fit(x)), shape="bernoulli")
    elif isinstance(x, Interval):
        return pba.bernoulli(io.mean(x))
    else:
        raise TypeError("Input data type not supported")


@makedist("beta")
def MMbeta(x: np.ndarray):
    m = x.mean()
    s = x.var()

    alpha = m * (m * (1 - m) / s - 1)
    beta_p = (1 - m) * (m * (1 - m) / s - 1)
    return sps.beta(alpha, beta_p)


@makedist("beta")
def mm_beta(mean, std):
    """from given moments, return a beta distribution"""
    m = mean
    s = std**2

    alpha = m * (m * (1 - m) / s - 1)
    beta_p = (1 - m) * (m * (1 - m) / s - 1)
    return sps.beta(alpha, beta_p)


@makedist("betabinom")
def MMbetabinomial(n: int, x):  # **
    #! 'x**2' variable repetition
    # n must be provided; it's not estimated from data
    # https://en.wikipedia.org/wiki/Beta-binomial_distribution#Example:
    # MMbetabinomial(n=12,rep(0:12,c(3,24,104,286,670,1033,1343,1112,829,478,181,45,7)))
    m1 = x.mean()
    m2 = (x**2).mean()
    d = n * (m2 / m1 - m1 - 1) + m1
    return sps.betabinom(n, (n * m1 - m2) / d, (n - m1) * (n - m2 / m1) / d)


@makedist("binom")
def MMbinomial(x, n: int):  # **
    #! the return seems overcomplicated?
    """
    args:
         - n (int): number of trials
    """
    return sps.binom(n, io.mean() / n)
    # a = x.mean()
    # b = x.std()
    # return binom(int(np.abs(np.round(a/(1-b**2/a)))), np.abs(1-b**2/a))


@makedist("chi2")
def MMchisquared(x):
    return sps.chi2(np.round(x.mean()))


# def MMchisquared(x):
#     #! TODO interval outward_rounding
#     if isinstance(x, sps.CensoredData | np.ndarray | list):
#         return D.dist_from_sps(chi2(outward_rounding(mean(x))), shape='chi2')
#     elif isinstance(x, Interval):
#         return pba.expon(mean(x))
#     else:
#         raise TypeError('Input data type not supported')


def MMexponential(x):
    import pyuncertainnumber.pba.intervals.intervalOperators as io

    mean_ = io.mean(x)
    return universal_exponential(mean_)


def universal_exponential(mean):
    """single parameter pattern exponential distribution constructor

    note:
        mean can be a precise number or an interval; Exponential is parameterised by scale, which is the inverse of the rate parameter (lambda).
    """
    if isinstance(mean, Number):
        return mm_exponential(mean)
    elif isinstance(mean, Interval):
        return pba.exponential(scale=mean)


@makedist("exponential")
def mm_exponential(mean):
    """from given moments, return an exponential distribution

    note:
        This constructor is parameterised by scale, which is the inverse of the rate parameter (lambda).
    """
    return sps.expon(scale=mean)  # scale is 1/lambda


@makedist("f")
def MMF(x):  # **
    w = 2 / (1 - 1 / x.mean())
    return sps.f(
        np.round(
            (2 * w**3 - 4 * w**2) / ((w - 2) ** 2 * (w - 4) * x.std() ** 2 - 2 * w**2)
        ),
        np.round(w),
    )


@makedist("gamma")
def MMgamma(x):  # **
    a = x.mean()
    b = x.std()
    return sps.gamma(b**2 / a, 1 / (a / b) ** 2)  # gamma1(a, b) ~ gamma(b²/a, (a/b)²)


@makedist("gamma")
def mm_gamma(mean, std=None):
    """from given moments, return a gamma distribution"""

    return sps.gamma(
        std**2 / mean, 1 / (mean / std) ** 2
    )  # gamma1(a, b) ~ gamma(b²/a, (a/b)²)


@makedist("geometric")
def MMgeometric(x):
    pass


def MMgeometric(x):
    if isinstance(x, sps.CensoredData | np.ndarray | list):
        return D.dist_from_sps(sps.geom(1 / (1 + io.mean(x))), shape="geometric")
    elif isinstance(x, Interval):
        return pba.geom(1 / (1 + io.mean(x)))
    else:
        raise TypeError("Input data type not supported")


def MMpascal(x):
    if isinstance(x, sps.CensoredData | np.ndarray | list):
        return D.dist_from_sps(sps.geom(1 / (1 + io.mean(x))), shape="pascal")
    elif isinstance(x, Interval):
        return pba.geom(1 / (1 + io.mean(x)))
    else:
        raise TypeError("Input data type not supported")


# def MMgumbel0(x) : return(gumbel(x.mean() - 0.57721 * x.std() * np.sqrt(6)/ np.pi, x.std() * np.sqrt(6)/ np.pi))       #**  # https://stackoverflow.com/questions/51427764/using-method-of-moments-with-gumbel-r-in-python-scipy-stats-gumbel-r


@makedist("gumbel")
def MMgumbel(x):  # **
    # https://stackoverflow.com/questions/51427764/using-method-of-moments-with-gumbel-r-in-python-scipy-stats-gumbel-r
    scale = np.sqrt(6) / np.pi * np.std(x)
    loc = np.mean(x) - np.euler_gamma * scale
    return sps.gumbel_r(loc, scale)


@makedist("gumbel")
def mm_gumbel(mean, std):
    """from given moments, return a gumbel distribution"""
    scale = np.sqrt(6) / np.pi * std
    loc = mean - np.euler_gamma * scale
    return sps.gumbel_r(loc, scale)


@makedist("extremevalue")
def MMextremevalue(x):
    return sps.gumbel_r(
        x.mean() - 0.57721 * x.std() * np.sqrt(6) / np.pi, x.std() * np.sqrt(6) / np.pi
    )  # **


# TODO possibly incorrect. See the function below.
@makedist("lognormal")
def MMlognormal(x):
    return sps.lognorm(x.mean(), x.std())  # **


def lognormal_mom_estimator(data=None, sample_mean=None, sample_variance=None):
    """
    Method of Moments estimator for Lognormal(μ, σ^2).

    Parameters
    ----------
    data : array-like, optional
        Raw data sample. If provided, mean and variance are computed from it.
    sample_mean : float, optional
        Precomputed sample mean (ignored if data is provided).
    sample_variance : float, optional
        Precomputed sample variance (ignored if data is provided).

    Returns
    -------
    mu_hat : float
        Estimated mean of the underlying normal distribution.
    sigma2_hat : float
        Estimated variance of the underlying normal distribution.
    """
    # If raw data is provided, compute mean and variance
    if data is not None:
        data = np.asarray(data)
        if np.any(data <= 0):
            raise ValueError(
                "All data points must be positive for lognormal distribution."
            )
        sample_mean = data.mean()
        sample_variance = data.var(ddof=0)  # population variance
    elif sample_mean is None or sample_variance is None:
        raise ValueError(
            "Provide either raw data or both sample_mean and sample_variance."
        )

    if sample_mean <= 0:
        raise ValueError("Sample mean must be positive for lognormal distribution.")

    # Method of moments formulas
    sigma2_hat = np.log(1 + sample_variance / (sample_mean**2))
    mu_hat = np.log(sample_mean) - sigma2_hat / 2

    return mu_hat, sigma2_hat


def mm_lognormal(mean: float, std: float):
    """
    Method of Moments estimator for Lognormal(μ, σ^2).

    args
        mean (float):
            Sample mean of the data (m1).
        std (float):
            Sample standard deviation of the data (s).

    returns
        mu_hat (float):
            Estimated mean of the underlying normal distribution.
        sigma2_hat (float):
            Estimated variance of the underlying normal distribution.
    """

    var = std**2

    if mean <= 0:
        raise ValueError("Sample mean must be positive for lognormal distribution.")

    # Estimate sigma^2
    sigma2_hat = np.log(1 + var / (mean**2))

    # Estimate mu
    mu_hat = np.log(mean) - sigma2_hat / 2

    return mu_hat, sigma2_hat


@makedist("laplace")
def MMlaplace(x):
    return sps.laplace(x.mean(), x.std() / np.sqrt(2))  # **


@makedist("doubleexponential")
def MMdoubleexponential(x):
    return sps.laplace(x.mean(), x.std() / np.sqrt(2))  # **


@makedist("logistic")
def MMlogistic(x):
    return sps.logistic(x.mean(), x.std() * np.sqrt(3) / np.pi)  # **


@makedist("loguniform")
def MMloguniform(x):
    return sps.loguniform(mean=x.mean(), std=x.std())  # **


@makedist("norm")
def MMnormal(x):
    return sps.norm(x.mean(), x.std())  # **


@makedist("norm")
def mm_normal(mean, std):
    """from given moments, return a normal distribution"""

    return sps.norm(loc=mean, scale=std)


@makedist("gaussian")
def MMgaussian(x):
    return sps.norm(x.mean(), x.std())  # **


@makedist("pareto")
def MMpareto(x):  # **
    a = x.mean()
    b = x.std()
    return sps.pareto(
        a / (1 + 1 / np.sqrt(1 + a**2 / b**2)), 1 + np.sqrt(1 + a**2 / b**2)
    )


def MMpoisson(x):
    shape = "poisson"
    if isinstance(x, sps.CensoredData | np.ndarray | list):
        return D.dist_from_sps(named_dists.get(shape)(io.mean(x)), shape=shape)
    elif isinstance(x, Interval):
        return named_pbox.get(shape)(io.mean(x))
    else:
        raise TypeError("Input data type not supported")


@makedist("powerlaw")
def MMpowerfunction(x):  # **
    a = x.mean()
    b = x.std()
    return sps.powerlaw(
        a / (1 - 1 / np.sqrt(1 + (a / b) ** 2)), np.sqrt(1 + (a / b) ** 2) - 1
    )


@makedist("t")
def MMt(x):
    return sps.t(2 / (1 - 1 / x.std() ** 2))


@makedist("student")
def MMstudent(x):
    assert not (1 < x.std()), "Improper standard deviation for student distribution"
    return sps.t(2 / (1 - 1 / x.std() ** 2))


@makedist("uniform")
def MMuniform(x):  # **
    a = x.mean()
    b = np.sqrt(3) * x.std()
    return sps.uniform(a - b, a + b)


@makedist("rectangular")
def MMrectangular(x):
    return MMuniform(x)


@makedist("triangular")
def MMtriangular(x, iters=100, dives=10):  # **
    # iterative search for triangular distribution parameters using method of
    # matching moments (you solve the thing analytically! too messy without help)
    # testing code indicated with #-#
    # -#some = 10
    # -#A = runif(1,0,10)
    # -#B = A + runif(1,0,10)
    # -#C = runif(1,A,B)
    # -#x = qtriangular(runif(some), A,C,B)
    def skewness(x):
        m = x.mean()
        # std uses the population formula, may need the sample formula
        return np.sum((x - m) ** 3) / ((len(x) - 1) * x.std() ** 3)

    M = np.mean(x)
    V = np.var(x)
    S = skewness(x)
    a = aa = min(x)  # apparently double assignments work
    b = bb = max(x)
    c = cc = 3 * M - a - b
    many = iters
    s1 = np.std(x)
    for k in range(dives):
        s1 = s2 = s3 = s1 / 2
        a = np.random.normal(aa, s1, many)
        b = np.random.normal(bb, s2, many)
        c = np.random.normal(cc, s3, many)
        m = (a + b + c) / 3
        k = a**2 + b**2 + c**2 - a * b - a * c - b * c
        v = k / 18
        s = (np.sqrt(2) * (a + b - 2 * c) * (2 * a - b - c) * (a - 2 * b + c)) / (
            5 * k ** (3 / 2)
        )
        d = (M - m) ** 2 + (V - v) ** 2 + (S - s) ** 2
        i = np.argmin(d)  # which.min(d)
        aa = a[i]
        bb = b[i]
        cc = c[i]
    # -#gray(triangular(A,B,C), new = TRUE)
    # -#blue(x)
    # -#green(triangular(aa,bb,cc))
    # -#A;aa; B;bb; C;cc  # the order is min, max, mode
    print(aa, bb, cc)
    return sps.triang(aa, cc, bb)


# no decorator needed herein
def MMrayleigh(x):
    return mm_rayleigh(x.mean())


def universal_rayleigh(mean):
    """single parameter pattern rayleigh distribution constructor

    note:
        mean can be a precise number or an interval;
    """
    if isinstance(mean, Number):
        return mm_rayleigh(mean)
    elif isinstance(mean, Interval):
        _scale = mean / np.sqrt(np.pi / 2)
        return pba.rayleigh(scale=_scale)


@makedist("rayleigh")
def mm_rayleigh(mean):
    """from given moments, return a rayleigh distribution"""
    return sps.rayleigh(scale=mean / np.sqrt(np.pi / 2))


named_mom_methods = {
    "bernoulli": MMbernoulli,
    "beta": MMbeta,
    "betabinomial": MMbetabinomial,
    "binomial": MMbinomial,
    "chisquared": MMchisquared,
    "exponential": MMexponential,
    "expon": MMexponential,
    "F": MMF,
    "f": MMF,
    "gamma": MMgamma,
    "geometric": MMgeometric,
    "gumbel": MMgumbel,
    "extremevalue": MMextremevalue,
    "lognormal": MMlognormal,
    "laplace": MMlaplace,
    "doubleexponential": MMdoubleexponential,
    "logistic": MMlogistic,
    "loguniform": MMloguniform,
    "norm": MMnormal,
    "normal": MMnormal,
    "gaussian": MMgaussian,
    "pareto": MMpareto,
    "poisson": MMpoisson,
    "powerfunction": MMpowerfunction,
    "rayleigh": MMrayleigh,
    "t": MMt,
    "student": MMstudent,
    "uniform": MMuniform,
    "rectangular": MMrectangular,
    "triangular": MMtriangular,
}

###############################################################################
# * Alternative maximum likelihood estimation constructors using scipy.stats  *#
###############################################################################

# Some of these functions may support intervals in the data x.  See
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.CensoredData.html#scipy.stats.CensoredData


@makedist("bernoulli")
def MLbernoulli(x):
    return sps.bernoulli(*sps.bernoulli.fit(x))


@makedist("beta")
def MLbeta(x):
    return sps.beta(*sps.beta.fit(x))


@makedist("betabinom")
def MLbetabinomial(x):
    return sps.betabinom(*sps.betabinom.fit(x))


@makedist("binom")
def MLbinomial(x):
    """
    # TODO to check
    #! no fitting func for scipy discrete distributions
    """
    return sps.binom(*sps.binom.fit(x))


@makedist("chi2")
def MLchisquared(x):
    return sps.chi2(*sps.chi2.fit(x))


@makedist("expon")
def MLexponential(x):
    """a standalone caller for exponential distribution with interval data (not in use yet)"""
    if isinstance(x, sps.CensoredData | np.ndarray | list):
        return sps.expon(*sps.expon.fit(x))


@makedist("f")
def MLF(x):
    return sps.f(*sps.f.fit(x))


@makedist("gamma")
def MLgamma(x):
    return sps.gamma(*sps.gamma.fit(x))


@makedist("gammaexpon")
def MLgammaexponential(x):
    return sps.gammaexpon(*sps.gammaexpon.fit(x))


@makedist("geom")
def MLgeometric(x):
    return sps.geom(*sps.geom.fit(x))


@makedist("gumbel")
def MLgumbel(x):
    return sps.gumbel_r(*sps.gumbel_r.fit(x))


@makedist("laplace")
def MLlaplace(x):
    return sps.laplace(*sps.laplace.fit(x))


@makedist("logistic")
def MLlogistic(x):
    return sps.logistic(*sps.logistic.fit(x))


@makedist("lognorm")
def MLlognormal(x):
    return sps.lognorm(*sps.lognorm.fit(x))


# TODO why not use `sps.lognorm.fit(x)`` directly?
# def MLlognormal(x): return (sps.lognorm.rvs(
#     *sps.lognorm.fit(x), size=many))


@makedist("loguniform")
def MLloguniform(x):
    return sps.loguniform(*sps.loguniform.fit(x))


@makedist("nbinom")
def MLnegativebinomial(x):
    return sps.nbinom(*sps.nbinom.fit(x))


@makedist("norm")
def MLnormal(x):
    return sps.norm(*sps.norm.fit(x))


@makedist("pareto")
def MLpareto(x):
    return sps.pareto(*sps.pareto.fit(x))


@makedist("poisson")
def MLpoisson(x):
    return sps.poisson(*sps.poisson.fit(x))


@makedist("powerlaw")
def MLpowerfunction(x):
    return sps.powerlaw(*sps.powerlaw.fit(x))


@makedist("rayleigh")
def MLrayleigh(x):
    return sps.rayleigh(*sps.rayleigh.fit(x))


@makedist("t")
def MLstudent(x):
    return sps.t(*sps.t.fit(x))


@makedist("triang")
def MLtriangular(x):
    return sps.triang(*sps.triang.fit(x))


@makedist("uniform")
def MLuniform(x):
    return sps.uniform(*sps.uniform.fit(x))


named_mle_methods = {
    "bernoulli": MLbernoulli,
    "beta": MLbeta,
    "betabinomial": MLbetabinomial,
    "binomial": MLbinomial,
    "chisquared": MLchisquared,
    "exponential": MLexponential,
    "expon": MLexponential,
    "F": MLF,
    "f": MLF,
    "gamma": MLgamma,
    "gammaexponential": MLgammaexponential,
    "geometric": MLgeometric,
    "gumbel": MLgumbel,
    "laplace": MLlaplace,
    "logistic": MLlogistic,
    "lognormal": MLlognormal,
    "loguniform": MLloguniform,
    "negativebinomial": MLnegativebinomial,
    "norm": MLnormal,
    "normal": MLnormal,
    "gaussian": MLnormal,
    "pareto": MLpareto,
    "poisson": MLpoisson,
    "powerfunction": MLpowerfunction,
    "rayleigh": MLrayleigh,
    "student": MLstudent,
    "triangular": MLtriangular,
    "uniform": MLuniform,
}


##########################################################################
# Scott's maximum likelihood estimation constructors
##########################################################################

# Some of these functions may not support intervals in the data x           #**


def sMLbernoulli(x):
    return sps.bernoulli(x.mean())


def sMLnormal(x):
    return sps.norm(x.mean(), x.sd())  # **


def sMLgaussian(x):
    return MLnormal(x)


def sMLexponential(x):
    return sps.expon(x.mean())


def sMLpoisson(x):
    return sps.poisson(x.mean())


def sMLgeometric(x):
    return sps.geom(1 / (1 + x.mean()))


def sMLgumbel(x):
    loc, scale = sps.gumbel_r.fit(x)
    return sps.gumbel_r(loc, scale)


def sMLpascal(x):
    return MLgeometric(x)


def sMLuniform(x):
    return sps.uniform(min(x), max(x))  # **


def sMLrectangular(x):
    return MLuniform(x)


def sMLpareto(x):
    return sps.pareto(min(x), len(x) / np.sum(np.log(x) - np.log(min(x))))  # **


def sMLlaplace(x):
    return sps.laplace(x.median(), np.sum(np.abs(x - x.median()) / len(x)))  # **


def sMLdoubleexponential(x):
    return MLlaplace(x)


def sMLlognormal2(x):  # **
    n = len(x)
    mu = np.sum(np.log(x)) / n
    # this function gives clearly poor results
    return sps.lognormal2(mlog=mu, slog=np.sum((np.log(x) - mu) ** 2) / n)


# just uses transformation, which seems unlikely to be true, but fitdistrplus package uses it too
def sMLlognormal(x):
    return np.exp(MLnormal(np.log(x)))


def sMLloguniform(x):
    a, b, _, _ = sps.loguniform.fit(x)
    return sps.loguniform(a, b)


def sMLweibull(x, shapeinterval=None):  # **
    if shapeinterval is None:
        shapeinterval = np.array((0.001, 500))

    def wf(k):
        return (
            np.sum(x**k * np.log(x)) / np.sum(x**k) - np.sum(np.log(x)) / len(x) - 1 / k
        )

    k = sps.uniroot(wf, shapeinterval)
    el = np.exp(np.log(np.sum(x**k) / len(x)) / k)
    return sps.weibull_min.rvs(scale=el, shape=k)


def sMLgamma(data):  # **
    xbar = data.mean()
    shape = (xbar / data.sd()) ** 2  # initial estimate of shape from MoM
    logxbar = np.log(xbar)
    meanlog = np.log(data).mean()

    def f(x):
        return np.log(x) - np.digamma(x) - logxbar + meanlog

    shape = sps.uniroot(f, shape * np.array((0.5, 1.5))).root()
    rate = shape / xbar
    return sps.gamma(shape=shape, rate=rate)


named_smle_methods = {
    "bernoulli": sMLbernoulli,
    "normal": sMLnormal,
    "gaussian": sMLgaussian,
    "exponential": sMLexponential,
    "poisson": sMLpoisson,
    "geometric": sMLgeometric,
    "gumbel": sMLgumbel,
    "pascal": sMLpascal,
    "uniform": sMLuniform,
    "rectangular": sMLrectangular,
    "pareto": sMLpareto,
    "laplace": sMLlaplace,
    "doubleexponential": sMLdoubleexponential,
    "lognormal2": sMLlognormal2,
    "lognormal": sMLlognormal,
    "loguniform": sMLloguniform,
    "weibull": sMLweibull,
    "gamma": sMLgamma,
}


##########################################################################
# Maximum entropy distribution constructors
##########################################################################


def MEminmax(min, max):
    return sps.uniform(min, max)


# http://mathoverflow.net/questions/116667/whats-the-maximum-entropy-probability-distribution-given-bounds-a-b-and-mean, http://www.math.uconn.edu/~kconrad/blurbs/analysis/entropypost.pdf for discussion of this solution.
def MEminmaxmean(min, max, mean):
    return sps.sawinconrad(min, mean, max)


def MEmeansd(mean, sd):
    return sps.normal(mean, sd)


def MEminmean(min, mean):
    return min + sps.exponential(mean - min)


def MEdiscretemean(x, mu, steps=10, iterations=50):  # e.g., MEdiscretemean(1:10,2.3)
    x = np.array(x)

    def fixc(x, r):
        return 1 / np.sum(r**x)

    r = br = 1
    c = bc = fixc(x, r)
    d = bd = (mu - np.sum((c * r**x) * x)) ** 2
    for j in range(steps):
        step = 1 / (j + 1)
        for i in range(iterations):
            r = np.abs(br + (np.random.uniform() - 0.5) * step)
            c = fixc(x, r)
            d = (mu - np.sum((c * r**x) * x)) ** 2
            if d < bd:
                br = r
                bc = c
                bd = d
    w = bc * br**x
    w = w / np.sum(w)  # needed?

    z = np.array([])
    k = len(x)
    for i in range(k):
        z = np.concatenate((z, np.repeat(x[i], w[i] * many)))
    if len(z) >= many:
        z = z[0:many]
    else:
        z = np.concatenate((z, np.random.choice(x, size=many - len(z), p=w)))
    np.random.shuffle(z)  # shuffles z in place
    return z


def MEquantiles(v, p):
    if len(v) != len(p):
        stop("Inconsistent array lengths for quantiles")
    if (min(p) < 0) or (1 < max(p)):
        stop("Improper probability for quantiles")  # ensure 0 <= p <= 1
    if not (min(p) == 0 and max(p) == 1):
        stop("Probabilities must start at zero and go to one for quantiles")
    if np.any(np.diff(p) < 0):
        # ensure montone probabilities
        stop("Probabilities must increase for quantiles")
    if np.any(np.diff(v) < 0):
        stop("Quantiles values must increase")  # ensure montone quantiles
    x = np.repeat(np.inf, many)
    r = np.random.uniform(size=many)
    # np.where is Python's version of R's ifelse function
    for i in range(len(p) - 1):
        x = np.where(
            (p[i] <= r) & (r < p[i + 1]),
            v[i] + (r - p[i]) * (v[i + 1] - v[i]) / (p[i + 1] - p[i]),
            x,
        )
    return x


def MEdiscreteminmax(min, max):
    return np.minimum(np.trunc(uniform(min, max + 1)), max)


def MEmeanvar(mean, var):
    return MEmeansd(mean, np.sqrt(var))


def MEminmaxmeansd(min, max, mean, sd):
    return beta1((mean - min) / (max - min), sd / (max - min)) * (max - min) + min


def MEmmms(min, max, mean, sd):
    return beta1((mean - min) / (max - min), sd / (max - min)) * (max - min) + min


def MEminmaxmeanvar(min, max, mean, var):
    return MEminmaxmeansd(min, max, mean, np.sqrt(var))


###############################################################################
# Miscellaneous: PERT, Fermi methods, mean-normal range, KS, EDF, Antweiler
###############################################################################


# https://wernerantweiler.ca/blog.php?item=2019-06-05       #**
def antweiler(x):
    return sps.triang(min=min(x), mode=3 * np.mean(x) - max(x) - min(x), max=max(x))


def betapert(min, max, mode):  # N.B.  Not in numerical order!
    mu = (min + max + 4 * mode) / 6
    if abs(mode - mu) < 1e-8:
        alpha1 = alpha2 = 3
    else:
        alpha1 = (mu - min) * (2 * mode - min - max) / ((mode - mu) * (max - min))
        alpha2 = alpha1 * (max - mu) / (mu - min)
    return min + (max - min) * sps.beta(alpha1, alpha2)


def mnr(n, many=10000):
    xL = xU = np.random.normal(size=many)
    for i in range(n - 1):
        xx = np.random.normal(size=many)
        xL = np.minimum(xL, xx)
        xU = np.maximum(xU, xx)
    return np.mean(xU - xL)


def fermilnorm(x1, x2, n=None, pr=0.9):
    gm = np.sqrt(x1 * x2)
    if n is None:
        gsd = np.sqrt(x2 / x1) ** (1 / sps.norm.ppf(pr))  # qnorm(pr)
    else:
        gsd = np.exp((np.log(x2) - np.log(x1)) / mnr(n))
    return np.log((gm, gsd))


def ferminorm(x1, x2, n=None, pr=0.9):
    m = (x1 + x2) / 2
    if n is None:
        s = (x2 - x1) / (2 * sps.norm.ppf(pr))  # qnorm(pr)
    else:
        s = (x2 - x1) / mnr(n)
    return np.array((m, s))


def approxksD95(n):
    from scipy.interpolate import CubicSpline

    # approximations for the critical level for Kolmogorov-Smirnov statistic D,
    # for confidence level 0.95. Taken from Bickel & Doksum, table IX, p.483
    # and Lienert G.A.(1975) who attributes to Miller,L.H.(1956), JASA
    if n > 80:
        # Bickel&Doksum, table IX,p.483
        return 1.358 / (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n))
    else:
        x = np.array(
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80)
        )  # from Lienert
        y = np.array(
            (
                0.975,
                0.84189,
                0.70760,
                0.62394,
                0.56328,  # 1:5
                0.51926,
                0.48342,
                0.45427,
                0.43001,
                0.40925,  # 6:10
                0.33760,
                0.29408,
                0.24170,
                0.21012,  # 15,20,30,40
                0.18841,
                0.17231,
                0.15975,
                0.14960,
            )
        )  # 50,60,70,80
        f = CubicSpline(x, y, bc_type="natural")
        return f(n)


def ks(x, conf=0.95, min=None, max=None):
    if conf != 0.95:
        stop("Cannot currently handle confidence levels other than 95%")
    h = histogram(x)
    mn = np.min(x)
    mx = np.max(x)
    if min is None:
        min = mn - (mx - mn) / 2
    if max is None:
        max = mx + (mx - mn) / 2
    lots = int(approxksD95(len(x)) * many)
    Lfermi = np.concatenate((lots * [min], h))
    Rfermi = np.concatenate((h, lots * [max]))
    Lfermi.sort()
    Rfermi[::-1].sort()
    # should prolly shuffle
    return np.concatenate((Lfermi[0:many], Rfermi[0:many]))


def ferminormconfband(x1, x2, n, pr=0.9, conf=0.95, bOt=0.001, tOp=0.999):
    if conf != 0.95:
        stop("Cannot handle confidence levels other than 95%")
    m, s = ferminorm(x1, x2, n, pr)
    lots = int(approxksD95(n) * many)
    BOT = scipy.stats.norm.ppf(bOt, m, s)
    TOP = scipy.stats.norm.ppf(tOp, m, s)
    Lfermi = np.concatenate((lots * [BOT], scipy.stats.norm.rvs(m, s, size=many)))
    Rfermi = np.concatenate((lots * [TOP], scipy.stats.norm.rvs(m, s, size=many)))
    Lfermi.sort()
    Rfermi[::-1].sort()
    # should prolly shuffle
    return np.concatenate((Lfermi[0:many], Rfermi[0:many]))


def fermilnormconfband(x1, x2, n, pr=0.9, conf=0.95, bOt=0.001, tOp=0.999):
    if conf != 0.95:
        stop("Cannot handle confidence levels other than 95%")
    mlog, slog = fermilnorm(x1, x2, n, pr)
    d = lognormal2(mlog, slog)
    lots = int(approxksD95(n) * many)
    BOT = scipy.stats.lognorm.ppf(bOt, s=slog, scale=np.exp(mlog))
    TOP = scipy.stats.lognorm.ppf(tOp, s=slog, scale=np.exp(mlog))
    Lfermi = np.concatenate((lots * [BOT], d))
    Rfermi = np.concatenate((lots * [TOP], d))
    Lfermi.sort()
    Rfermi[::-1].sort()
    # should prolly shuffle
    return np.concatenate((Lfermi[0:many], Rfermi[0:many]))


# * --------------- moments shape distribution constructors --------------- *#


# TODO: add more distributions in the future


def parse_moments(
    family: str,
    mean: Number | Interval,
    std: Number = None,
    var: Number = None,
    **kwargs,
):
    """parse the moments input to a standard form for dist construction

    args:
        family (str): distribution family
        mean (Number | Interval): mean value, which could be either precise value or an Interval object. Python list is not supported.
        std (Number): standard deviation
        var (Number): variance

    example:
        >>> parse_moments(family='normal', mean=3., std=1.)

    note:
        Only accept up to 2nd moment for now. Interval mean is supported due to single parameter construction.
        Combined mean and std intervals are not supported as they are deemed NP hard question.
    """

    # var --> std
    # Enforce that exactly one of std or var is given
    if std is not None and var is not None:
        raise ValueError("You cannot provide boh 'std' or 'var'.")

    # If variance is provided, convert to std
    if var is not None:
        try:
            std = np.sqrt(var)
        except:
            raise ValueError("Variance must be non-negative.")

    match family:
        # 1 parameter distributions
        case "exponential":
            return universal_exponential(mean=mean)
        case "rayleigh":
            return universal_rayleigh(mean=mean)
        # 2 parameter distributions
        case "normal" | "gaussian":
            return mm_normal(mean=mean, std=std)
        case "gamma":
            return mm_gamma(mean=mean, std=std)
        case "beta":
            return mm_beta(mean=mean, std=std)
        case "gumbel":
            return mm_gumbel(mean=mean, std=std)
        case "lognormal":
            return mm_lognormal(mean=mean, std=std)
        case _:
            raise ValueError(f"distribution family {family} not yet supported")
