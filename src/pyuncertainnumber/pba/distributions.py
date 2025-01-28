"""distribution constructs """

import numpy as np
import scipy.stats as sps
import matplotlib as mpl
from warnings import *
from dataclasses import dataclass
from typing import *
from ..characterisation.utils import pl_pcdf, pl_ecdf
import scipy
from .params import Params
from .pbox import named_pbox


@dataclass
class Distribution:
    """two signature for the distribution object, either a parametric specification or a nonparametric sample per se"""

    dist_family: str = None
    dist_params: list[float] | Tuple[float, ...] = None
    sample_data: list[float] | np.ndarray = None

    def __post_init__(self):
        if all(
            v is None for v in [self.dist_family, self.dist_params, self.sample_data]
        ):
            raise ValueError(
                "At least one of dist_family, dist_params or sample must be specified"
            )
        self.flag()
        self._dist = self.rep()
        self.make_naked_value()

    def __repr__(self):
        # if self.sample_data is not None:
        #     return "sample-approximated distribution object"
        if self.dist_params is not None:
            return f"dist ~ {self.dist_family}{self.dist_params}"
        elif self.sample_data is not None:
            return "dist ~ sample-approximated distribution object"
        else:
            return "wrong initialisation"

    def rep(self):
        """the dist object either sps dist or sample approximated or pbox dist"""
        if self.dist_family is not None:
            return named_dists.get(self.dist_family)(*self.dist_params)

    def flag(self):
        """boolean flag for if the distribution is a parameterised distribution or not
        note:
            - only parameterised dist can do sampling
            - for non-parameterised sample-data based dist, next steps could be fitting
        """
        if (self.dist_params is not None) & (self.dist_family is not None):
            self._flag = True
        else:
            self._flag = False

    def sample(self, size):
        """generate deviates from the distribution"""
        if self._flag:
            return self._dist.rvs(size=size)
        else:
            raise ValueError(
                "Sampling not supported for sample-approximated distributions"
            )

    def make_naked_value(self):
        """one value representation of the distribution
        note:
            - use mean for now;
        """
        if self._flag:
            self._naked_value = self._dist.mean()
        else:
            self._naked_value = np.mean(self.sample_data)

    def display(self, **kwargs):
        """display the distribution"""
        if self.sample_data is not None:
            return pl_ecdf(self.sample_data, **kwargs)
        pl_pcdf(self._dist, **kwargs)

    def _get_hint(self):
        pass

    def fit(self, data):
        """fit the distribution to the data"""
        pass

    @property
    def naked_value(self):
        return self._naked_value

    @property
    def hint(self):
        pass

    # *  ---------------------constructors---------------------* #
    @classmethod
    def dist_from_sps(
        cls, dist: sps.rv_continuous | sps.rv_discrete, shape: str = None
    ):
        params = dist.args + tuple(dist.kwds.values())
        return cls(dist_family=shape, dist_params=params)

    # *  ---------------------conversion---------------------* #

    def to_pbox(self):
        """convert the distribution to a pbox
        note:
            - this only works for parameteried distributions for now
            - later on work with sample-approximated dist until `fit()`is implemented
        """
        if self._flag:
            # pass
            return named_pbox.get(self.dist_family)(*self.dist_params)


# * ------------------ sample-approximated dist representation  ------------------ *#


def bernoulli(p):
    return np.random.uniform(size=Params.many) < p


def beta(a, b):
    # if (a==0) and (b==0) : return(env(np.repeat(0.0, Params.many), np.repeat(1.0, Params.many)))  # this is [0,1]
    if (a == 0) and (b == 0):
        return bernoulli(0.5)  # or should it be [0,1]?
    if a == 0:
        return np.repeat(0.0, Params.many)
    if b == 0:
        return np.repeat(1.0, Params.many)
    return scipy.stats.beta.rvs(a, b, size=Params.many)


def betabinomial2(size, v, w):
    return scipy.stats.binom.rvs(size, beta(v, w), size=Params.many)


def betabinomial(size, v, w):
    return scipy.stats.betabinom.rvs(size, v, w, size=Params.many)


def binomial(size, p):
    return scipy.stats.binom.rvs(size, p, size=Params.many)


def chisquared(v):
    return scipy.stats.chi2.rvs(v, size=Params.many)


def delta(a):
    return np.repeat(a, Params.many)


def exponential(rate=1, mean=None):
    if mean is None:
        mean = 1 / rate
    # rate = 1/mean
    return scipy.stats.expon.rvs(scale=mean, size=Params.many)


def exponential1(mean=1):
    return scipy.stats.expon.rvs(scale=mean, size=Params.many)


def F(df1, df2):
    return scipy.stats.f.rvs(df1, df2, size=Params.many)


def gamma(shape, rate=1, scale=None):
    if scale is None:
        scale = 1 / rate
    rate = 1 / scale
    return scipy.stats.gamma.rvs(a=shape, scale=1 / rate, size=Params.many)


def gammaexponential(shape, rate=1, scale=None):
    if scale is None:
        scale = 1 / rate
    rate = 1 / scale
    # expon(scale=gamma(a=shape, scale=1/rate))
    return scipy.stats.expon.rvs(
        scale=1 / scipy.stats.gamma.rvs(a=shape, scale=scale, size=Params.many),
        size=Params.many,
    )


def geometric(m):
    return scipy.stats.geom.rvs(m, size=Params.many)


def gumbel(loc, scale):
    return scipy.stats.gumbel_r.rvs(loc, scale, size=Params.many)


def inversechisquared(v):
    return 1 / chisquared(v)


def inversegamma(shape, scale=None, rate=None):
    if scale is None and not rate is None:
        scale = 1 / rate
    return scipy.stats.invgamma.rvs(a=shape, scale=scale, size=Params.many)


def laplace(a, b):
    return scipy.stats.laplace.rvs(a, b, size=Params.many)


def logistic(loc, scale):
    return scipy.stats.logistic.rvs(loc, scale, size=Params.many)


def lognormal(m, s):
    m2 = m**2
    s2 = s**2
    mlog = np.log(m2 / np.sqrt(m2 + s2))
    slog = np.sqrt(np.log((m2 + s2) / m2))
    return scipy.stats.lognorm.rvs(s=slog, scale=np.exp(mlog), size=Params.many)


def lognormal2(mlog, slog):
    return scipy.stats.lognorm.rvs(s=slog, scale=np.exp(mlog), size=Params.many)


# lognormal = function(mean=NULL, std=NULL, meanlog=NULL, stdlog=NULL, median=NULL, cv=NULL, name='', ...){
#  if (is.null(meanlog) & !is.null(median)) meanlog = log(median)
#  if (is.null(stdlog) & !is.null(cv)) stdlog = sqrt(log(cv^2 + 1))
#  # lognormal(a, b) ~ lognormal2(log(a^2/sqrt(a^2+b^2)),sqrt(log((a^2+b^2)/a^2)))
#  if (is.null(meanlog) & (!is.null(mean)) & (!is.null(std))) meanlog = log(mean^2/sqrt(mean^2+std^2))
#  if (is.null(stdlog) & !is.null(mean) & !is.null(std)) stdlog = sqrt(log((mean^2+std^2)/mean^2))
#  if (!is.null(meanlog) & !is.null(stdlog)) Slognormal0(meanlog,stdlog,name) else stop('not enough information to specify the lognormal distribution')
#  }


def loguniform_solve(m, v):
    def loguniform_f(a, m, v):
        return a * m * np.exp(2 * (v / (m**2) + 1)) + np.exp(2 * a / m) * (
            a * m - 2 * ((m**2) + v)
        )

    def LUgrid(aa, w):
        return left(aa) + (right(aa) - left(aa)) * w / 100.0

    aa = (m - np.sqrt(4 * v), m)  # interval
    a = m
    ss = loguniform_f(a, m, v)
    for j in range(4):
        for i in range(101):  # 0:100
            a = LUgrid(aa, i)
            s = abs(loguniform_f(a, m, v))
            if s < ss:
                ss = s
                si = i
        a = LUgrid(aa, si)
        aa = (LUgrid(aa, si - 1), LUgrid(aa, si + 1))  # interval
    return a


def loguniform(min=None, max=None, minlog=None, maxlog=None, mean=None, std=None):
    if (min is None) and (not (minlog is None)):
        min = np.exp(minlog)
    if (max is None) and (not (maxlog is None)):
        max = np.exp(maxlog)
    if (
        (max is None)
        and (not (mean is None))
        and (not (std is None))
        and (not (min is None))
    ):
        max = 2 * (mean**2 + std**2) / mean - min
    if (min is None) and (max is None) and (not (mean is None)) and (not (std is None)):
        min = loguniform_solve(mean, std**2)
        max = 2 * (mean**2 + std**2) / mean - min
    return scipy.stats.loguniform.rvs(min, max, size=Params.many)


def loguniform1(m, s):
    return loguniform(mean=m, std=s)


def negativebinomial(size, prob):
    return scipy.stats.nbinom.rvs(size, prob, size=Params.many)


def normal(m, s):
    return scipy.stats.norm.rvs(m, s, size=Params.many)


def pareto(mode, c):
    return scipy.stats.pareto.rvs(c, scale=mode, size=Params.many)


def poisson(m):
    return scipy.stats.poisson.rvs(m, size=Params.many)


def powerfunction(b, c):
    return scipy.stats.powerlaw.rvs(c, scale=b, size=Params.many)


# parameterisation of rayleigh differs from that in pba.r


def rayleigh(loc, scale):
    return scipy.stats.rayleigh.rvs(loc, scale, size=Params.many)


def sawinconrad(min, mu, max):  # WHAT are the 'implicit constraints' doing?
    def sawinconradalpha01(mu):
        def f(alpha):
            return 1 / (1 - 1 / np.exp(alpha)) - 1 / alpha - mu

        if np.abs(mu - 0.5) < 0.000001:
            return 0
        return uniroot(f, np.array((-500, 500)))

    def qsawinconrad(p, min, mu, max):
        alpha = sawinconradalpha01((mu - min) / (max - min))
        if np.abs(alpha) < 0.000001:
            return min + (max - min) * p
        else:
            min + (max - min) * ((np.log(1 + p * (np.exp(alpha) - 1))) / alpha)

    a = left(min)
    b = right(max)
    c = left(mu)
    d = right(mu)
    if c < a:
        c = a  # implicit constraints
    if b < d:
        d = b
    # return(qsawinconrad(np.random.uniform(size=Params.many), min, mu, max))
    return qsawinconrad(np.random.uniform(size=Params.many), min, mu, max)


def student(v):
    return scipy.stats.t.rvs(v, size=Params.many)


def uniform(a, b):
    return scipy.stats.uniform.rvs(
        a, b - a, size=Params.many
    )  # who parameterizes like this?!?!


def triangular(min, mode, max):
    return np.random.triangular(
        min, mode, max, size=Params.many
    )  # cheating: uses random rather than scipy.stats


def histogram(x):
    return x[(np.trunc(scipy.stats.uniform.rvs(size=Params.many) * len(x))).astype(int)]


def mixture(x, w=None):
    if w is None:
        w = np.repeat(1, len(x))
    print(Params.many)
    r = np.sort(scipy.stats.uniform.rvs(size=Params.many))[::-1]
    x = np.concatenate(([x[0]], x))
    w = np.cumsum(np.concatenate(([0], w))) / np.sum(w)
    u = []
    j = len(x) - 1
    for p in r:
        while True:
            if w[j] <= p:
                break
            j = j - 1
        u = np.concatenate(([x[j + 1]], u))
    return u[np.argsort(scipy.stats.uniform.rvs(size=len(u)))]


# * ---------------------Scott ancillary funcs --------------------- *#


def left(x):
    return np.min(x)


def right(x):
    return np.max(x)


def uniroot(f, a):
    # https://stackoverflow.com/questions/43271440/find-a-root-of-a-function-in-a-given-range
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/optimize.html#root-finding
    #    from scipy.optimize import brentq
    #    return(brentq(f, min(a), max(a))) #,args=(t0)) # any function arguments beyond the varied parameter
    from scipy.optimize import fsolve

    return fsolve(f, (min(a) + max(a)) / 2)


# a dict that links ''distribution name'' requiring specification to the scipy.stats distribution
named_dists = {
    "alpha": sps.alpha,
    "anglit": sps.anglit,
    "arcsine": sps.arcsine,
    "argus": sps.argus,
    "beta": sps.beta,
    "betaprime": sps.betaprime,
    "bradford": sps.bradford,
    "burr": sps.burr,
    "burr12": sps.burr12,
    "cauchy": sps.cauchy,
    "chi": sps.chi,
    "chi2": sps.chi2,
    "cosine": sps.cosine,
    "crystalball": sps.crystalball,
    "dgamma": sps.dgamma,
    "dweibull": sps.dweibull,
    "erlang": sps.erlang,
    "expon": sps.expon,
    "exponnorm": sps.exponnorm,
    "exponweib": sps.exponweib,
    "exponpow": sps.exponpow,
    "f": sps.f,
    "fatiguelife": sps.fatiguelife,
    "fisk": sps.fisk,
    "foldcauchy": sps.foldcauchy,
    "foldnorm": sps.foldnorm,
    # 'frechet_r' : sps.frechet_r,
    # 'frechet_l' : sps.frechet_l,
    "genlogistic": sps.genlogistic,
    "gennorm": sps.gennorm,
    "genpareto": sps.genpareto,
    "genexpon": sps.genexpon,
    "genextreme": sps.genextreme,
    "gausshyper": sps.gausshyper,
    "gamma": sps.gamma,
    "gengamma": sps.gengamma,
    "genhalflogistic": sps.genhalflogistic,
    "geninvgauss": sps.geninvgauss,
    # 'gibrat' : sps.gibrat,
    "gompertz": sps.gompertz,
    "gumbel_r": sps.gumbel_r,
    "gumbel_l": sps.gumbel_l,
    "halfcauchy": sps.halfcauchy,
    "halflogistic": sps.halflogistic,
    "halfnorm": sps.halfnorm,
    "halfgennorm": sps.halfgennorm,
    "hypsecant": sps.hypsecant,
    "invgamma": sps.invgamma,
    "invgauss": sps.invgauss,
    "invweibull": sps.invweibull,
    "johnsonsb": sps.johnsonsb,
    "johnsonsu": sps.johnsonsu,
    "kappa4": sps.kappa4,
    "kappa3": sps.kappa3,
    "ksone": sps.ksone,
    "kstwobign": sps.kstwobign,
    "laplace": sps.laplace,
    "levy": sps.levy,
    "levy_l": sps.levy_l,
    "levy_stable": sps.levy_stable,
    "logistic": sps.logistic,
    "loggamma": sps.loggamma,
    "loglaplace": sps.loglaplace,
    "lognorm": sps.lognorm,
    "loguniform": sps.loguniform,
    "lomax": sps.lomax,
    "maxwell": sps.maxwell,
    "mielke": sps.mielke,
    "moyal": sps.moyal,
    "nakagami": sps.nakagami,
    "ncx2": sps.ncx2,
    "ncf": sps.ncf,
    "nct": sps.nct,
    "norm": sps.norm,
    "gaussian": sps.norm,
    "norminvgauss": sps.norminvgauss,
    "pareto": sps.pareto,
    "pearson3": sps.pearson3,
    "powerlaw": sps.powerlaw,
    "powerlognorm": sps.powerlognorm,
    "powernorm": sps.powernorm,
    "rdist": sps.rdist,
    "rayleigh": sps.rayleigh,
    "rice": sps.rice,
    "recipinvgauss": sps.recipinvgauss,
    "semicircular": sps.semicircular,
    "skewnorm": sps.skewnorm,
    "t": sps.t,
    "trapz": sps.trapz,
    "triang": sps.triang,
    "truncexpon": sps.truncexpon,
    "truncnorm": sps.truncnorm,
    "tukeylambda": sps.tukeylambda,
    "uniform": sps.uniform,
    "vonmises": sps.vonmises,
    "vonmises_line": sps.vonmises_line,
    "wald": sps.wald,
    "weibull_min": sps.weibull_min,
    "weibull_max": sps.weibull_max,
    "wrapcauchy": sps.wrapcauchy,
    "bernoulli": sps.bernoulli,
    "betabinom": sps.betabinom,
    "binom": sps.binom,
    "boltzmann": sps.boltzmann,
    "dlaplace": sps.dlaplace,
    "geom": sps.geom,
    "hypergeom": sps.hypergeom,
    "logser": sps.logser,
    "nbinom": sps.nbinom,
    "planck": sps.planck,
    "poisson": sps.poisson,
    "randint": sps.randint,
    "skellam": sps.skellam,
    "zipf": sps.zipf,
    "yulesimon": sps.yulesimon,
}
