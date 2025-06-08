"""distribution constructs"""

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from warnings import *
from dataclasses import dataclass
from typing import *
from ..characterisation.utils import pl_pcdf, pl_ecdf
from .params import Params
from .pbox_parametric import named_pbox
import statsmodels.distributions.copula as Copula
from .dependency import Dependency
from statsmodels.distributions.copula.api import CopulaDistribution

# * --------------------- parametric cases --------------------- *#


@dataclass
class Distribution:
    """two signature for the distribution object, either a parametric specification or from a nonparametric empirical data set"""

    dist_family: str = None
    dist_params: list[float] | Tuple[float, ...] = None
    empirical_data: list[float] | np.ndarray = None

    def __post_init__(self):
        if all(
            v is None for v in [self.dist_family, self.dist_params, self.empirical_data]
        ):
            raise ValueError(
                "At least one of dist_family, dist_params or sample must be specified"
            )
        self.flag()
        self._dist = self.rep()
        self.make_naked_value()

    def __repr__(self):
        # if self.empirical_data is not None:
        #     return "sample-approximated distribution object"
        if self.dist_params is not None:
            return f"dist ~ {self.dist_family}{self.dist_params}"
        elif self.empirical_data is not None:
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

    def alpha_cut(self, alpha):
        """alpha cut interface"""
        return self._dist.ppf(alpha)

    def make_naked_value(self):
        """one value representation of the distribution
        note:
            - use mean for now;
        """
        if self._flag:
            self._naked_value = self._dist.mean()
        else:
            self._naked_value = np.mean(self.empirical_data)

    def plot(self, **kwargs):
        """display the distribution"""
        if self.empirical_data is not None:
            return pl_ecdf(self.empirical_data, **kwargs)
        pl_pcdf(self._dist, **kwargs)

    def display(self, **kwargs):
        self.plot(**kwargs)
        plt.show()

    def _get_hint(self):
        pass

    def fit(self, data):
        """fit the distribution to the data"""
        pass

    @property
    def naked_value(self):
        return np.round(self._naked_value, 3)

    @property
    def low(self):
        return self._dist.ppf(Params.p_lboundary)

    @property
    def hi(self):
        return self._dist.ppf(Params.p_hboundary)

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

    def __neg__(self):
        return -self.to_pbox()

    def __add__(self, other):
        p = self.to_pbox()
        return p.add(other, dependency="f")

    def __radd__(self, other):
        return self.add(other, dependency="f")

    def __sub__(self, other):
        p = self.to_pbox()
        return p.sub(other, dependency="f")

    def __rsub__(self, other):
        self = -self
        return self.add(other, dependency="f")

    def __mul__(self, other):
        p = self.to_pbox()
        return p.mul(other, dependency="f")

    def __rmul__(self, other):
        return self.mul(other, dependency="f")

    def __truediv__(self, other):
        p = self.to_pbox()
        return p.div(other, dependency="f")

    def __rtruediv__(self, other):
        p = self.to_pbox()
        try:
            return other * p.recip()
        except:
            return NotImplemented

    def __pow__(self, other):
        p = self.to_pbox()
        return p.pow(other, dependency="f")

    def __rpow__(self, other):
        if not hasattr(other, "__iter__"):
            other = np.array((other))
        p = self.to_pbox()
        return p.pow(other, dependency="f")


class JointDistribution:

    def __init__(
        self,
        copula: Dependency,
        marginals: list[Distribution],
    ):
        self.marginals = marginals
        self.copula = copula
        self._joint_dist = CopulaDistribution(
            copula=self.copula._copula, marginals=[m._dist for m in self.marginals]
        )
        self.ndim = len(self.marginals)

    @staticmethod
    def from_sps(copula: Copula, marginals: list[sps.rv_continuous]):
        return CopulaDistribution(copula=copula, marginals=marginals)

    def sample(self, size):
        """generate orginal-space samples from the joint distribution"""
        return self._joint_dist.rvs(size)

    def u_sample(self, size):
        """generate copula-space samples from the joint distribution"""
        return self.copula.rvs(size)


# * --------------------- non-parametric ecdf cases --------------------- *#


@dataclass
class eeCDF_bundle:
    """a handy tuple of eCDF function q and p"""

    quantiles: np.ndarray
    probabilities: np.ndarray
    # TODO plot ecdf not starting from 0

    @classmethod
    def from_sps_ecdf(cls, e):
        """utility to tranform sps.ecdf to eCDF_bundle"""
        return cls(e.cdf.quantiles, e.cdf.probabilities)

    def plot_bounds(self, other):
        """plot the lower and upper bounds"""
        return plot_two_eCDF_bundle(self, other)


def transform_eeCDF_bundle(e):
    """utility to tranform sps.ecdf to eCDF_bundle"""
    return eCDF_bundle(e.cdf.quantiles, e.cdf.probabilities)


def pl_ecdf_bounds_2(q1, p1, q2, p2, ax=None, marker="+"):
    """plot the bounding cdf functions with two sets of quantiles and probabilities"""
    if ax is None:
        fig, ax = plt.subplots()

    ax.step(q1, p1, marker=marker, c="g", where="post")
    ax.step(q2, p2, marker=marker, c="b", where="post")
    ax.plot([q1[0], q2[0]], [0, 0], c="b")
    ax.plot([q1[-1], q2[-1]], [1, 1], c="g")
    return ax


def plot_two_eCDF_bundle(cdf1, cdf2, ax=None, **kwargs):
    """plot two eCDF_bundle objects"""
    if ax is None:
        fig, ax = plt.subplots()
    q1, p1 = cdf1.quantiles, cdf1.probabilities
    q2, p2 = cdf2.quantiles, cdf2.probabilities
    return pl_ecdf_bounds_2(q1, p1, q2, p2, ax=ax, **kwargs)


def pl_ecdf_bounding_bundles(
    b_l: eCDF_bundle,
    b_r: eCDF_bundle,
    ax=None,
    legend=True,
    title=None,
    sig_level=None,
    bound_colors=None,
    label=None,
    alpha=None,
    linestyle=None,
    linewidth=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    def set_if_not_none(d, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                d[k] = v

    plot_bound_colors = bound_colors if bound_colors is not None else ["g", "b"]

    cdf_kwargs = {"drawstyle": "steps-post"}

    set_if_not_none(
        cdf_kwargs,
        label=label,
        linestyle=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )

    ax.plot(
        b_l.quantiles,
        b_l.probabilities,
        label=label if label is not None else f"KS condidence bands {sig_level}\% ",
        color=plot_bound_colors[0],
        **cdf_kwargs,
    )
    ax.plot(
        b_r.quantiles,
        b_r.probabilities,
        color=plot_bound_colors[1],
        **cdf_kwargs,
    )
    ax.plot(
        [b_l.quantiles[0], b_r.quantiles[0]],
        [0, 0],
        color=plot_bound_colors[1],
        **cdf_kwargs,
    )
    ax.plot(
        [b_l.quantiles[-1], b_r.quantiles[-1]],
        [1, 1],
        color=plot_bound_colors[0],
        **cdf_kwargs,
    )

    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend()


def ecdf(d):
    """return the quantile and probability of a ecdf

    note:
        Scott's version which leads to doubling the length of quantiles and probabilities
        to make it a step function
    """
    d = np.array(d)
    N = d.size
    pp = np.concatenate((np.arange(N), np.arange(1, N + 1))) / N
    dd = np.concatenate((d, d))
    dd.sort()
    pp.sort()
    return dd, pp


def weighted_ecdf(s, w=None, display=False) -> tuple:
    """compute the weighted ecdf from (precise) sample data

    args:
        s (array-like) : precise sample data
        w (array-like) : weights

    note:
        - Sudret eq.1

    return:
        ecdf in the form of a tuple of q and p
    """

    if w is None:
        # weights
        N = len(s)
        w = np.repeat(1 / N, N)
    else:
        w = np.array(w)

    # s, w = sorting(s, w)
    arr = np.stack((s, w), axis=1)
    arr = arr[np.argsort(arr[:, 0])]

    p = np.cumsum(arr[:, 1])

    # for box plotting
    q = np.insert(arr[:, 0], 0, arr[0, 0], axis=0)
    p = np.insert(p, 0, 0.0, axis=0)

    if display == True:
        fig, ax = plt.subplots()
        ax.step(q, p, marker="+", where="post")

    # return quantile and probabilities
    return q, p


# * ------------------ special sane cases ------------------ *#
def uniform_sane(a, b):
    return sps.uniform(loc=a, scale=b - a)


def lognormal_sane(mu, sigma):
    """The sane lognormal which creates a lognormal distribution object based on the mean (mu) and standard deviation (sigma)
    of the underlying normal distribution.

    args:
        - mu (float): Mean of the underlying normal distribution
        - sigma (float): Standard deviation of the underlying normal distribution

    Returns:
        - A scipy.stats.lognorm frozen distribution object
    """
    shape = sigma  # shape parameter for lognorm
    scale = np.exp(mu)  # scale parameter is exp(mu)
    return sps.lognorm(s=shape, scale=scale)


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
    # "lognorm": sps.lognorm,
    "lognormal": lognormal_sane,
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
    "normal": sps.norm,
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
    "uniform": uniform_sane,
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
