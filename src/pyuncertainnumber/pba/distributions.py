from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from warnings import *
from dataclasses import dataclass, field
from ..characterisation.utils import pl_pcdf, pl_ecdf
from .params import Params
from .pbox_parametric import named_pbox
import statsmodels.distributions.copula as Copula
from .dependency import Dependency
from statsmodels.distributions.copula.api import CopulaDistribution
from .pbox_abc import Staircase
from .ecdf import get_ecdf
from numbers import Number
from .mixins import NominalValueMixin
from numpy.typing import ArrayLike


if TYPE_CHECKING:
    from pyuncertainnumber import Interval

"""distribution constructs"""


# * --------------------- parametric cases --------------------- *#


@dataclass
class Distribution(NominalValueMixin):
    """Two signatures are currentlly supported, either a parametric specification or from a nonparametric empirical data set.

    note:
        the nonparametric instasntiation via arrtribute `empirical_data` will be deprecated soon.
        We have introduced a :class:`distributions.ECDF` class instead.

    example:
        >>> d = Distribution('gaussian', (0,1))
    """

    dist_family: str = None
    dist_params: list[float] | tuple[float, ...] = None
    empirical_data: list[float] | np.ndarray = None
    skip_post: bool = False

    def __post_init__(self):
        if self.skip_post:
            return
        elif all(
            v is None for v in [self.dist_family, self.dist_params, self.empirical_data]
        ):
            raise ValueError(
                "At least one of dist_family, dist_params or sample must be specified"
            )
        self.dist = self.rep()

    def __repr__(self):
        # if self.empirical_data is not None:
        #     return "sample-approximated distribution object"
        if self.dist_params is not None:
            return f"dist ~ {self.dist_family}{self.dist_params}"
        elif self.empirical_data is not None:
            return "dist ~ sample-approximated distribution object"
        else:
            return "blank object"

    def rep(self):
        """Create the underling dist object either sps dist or sample approximated or pbox dist

        note:
            underlying constructor to create the scipy.stats distribution object
        """
        if self.dist_family is not None:
            return self._match_distribution()

    #! **kwargs from sps.dist will yield error
    def _match_distribution(self):
        """match the distribution object based on the family and parameters"""
        params = self.dist_params
        if not isinstance(params, (tuple, list)):
            params = (params,)

        # TODO: special cases make it general
        if self.dist_family == "lognormal":  # special case for lognormal distribution
            return lognormal_sane(*params)
        return named_dists.get(self.dist_family)(*params)

    def parse_params_from_dist(self):
        self.dist_params = tuple(self._dist.args) + tuple(
            v for k, v in sorted(self._dist.kwds.items())
        )

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

    def generate_rns(self, N):
        """generate 'N' random numbers from the distribution"""
        return self.sample(N)

    def alpha_cut(self, alpha):
        """alpha cut interface"""
        return self._dist.ppf(alpha)

    def make_nominal_value(self):
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

    def get_PI(self, alpha: Number = 0.95) -> Interval:
        """Compute the predictive interval at the coverage level of `alpha`

        args:
            - alpha (float): coverage level, default is 0.95

        example:
            >>> from pyuncertainnumber import pba
            >>> d = pba.Distribution('gaussian', (0, 1))
            >>> pi = d.get_PI(alpha=0.95)
            >>> print(pi)  # prints the interval at the 95% coverage level
        """

        from pyuncertainnumber import Interval

        lo_cut_level = (1 - alpha) / 2  # 0.025
        hi_cut_level = 1 - lo_cut_level  # 0.975

        hi = self.alpha_cut(hi_cut_level)
        lo = self.alpha_cut(lo_cut_level)
        return Interval(lo, hi)

    def pdf(self, x: ArrayLike):
        """compute the probability density function (pdf) at x"""
        return self._dist.pdf(x)

    def log_pdf_eval(self, x: ArrayLike):
        """compute the log of probability density function (pdf) at x"""
        return self._dist.logpdf(x)

    def cdf(self, x: ArrayLike):
        """compute the cumulative distribution function (cdf) at x"""
        return self._dist.cdf(x)

    def get_whole_cdf(self):
        """return the cumulative distribution function (cdf)"""
        return self._dist.cdf(Params.p_values)

    def _compute_nominal_value(self):
        return np.round(self._naked_value, 3)

    @property
    def dist(self):
        """the underlying sps.dist object"""
        return self._dist

    @dist.setter
    def dist(self, value):
        self._dist = value
        if self.dist_params is None:
            self.parse_params_from_dist()
        self.flag()
        self.make_nominal_value()

    @property
    def lo(self):
        return self._dist.ppf(Params.p_lboundary)

    @property
    def hi(self):
        return self._dist.ppf(Params.p_hboundary)

    @property
    def range(self):
        from pyuncertainnumber import Interval

        return Interval(self.lo, self.hi)

    @property
    def hint(self):
        pass

    # *  ---------------------constructors---------------------* #
    @classmethod
    def dist_from_sps(
        cls, dist: sps.rv_continuous | sps.rv_discrete, shape: str = None
    ):
        # old version but does not work with 'scales'
        # params = dist.args + tuple(dist.kwds.values())
        # return cls(dist_family=shape, dist_params=params)
        # params = {"args": dist.args, "kwds": dist.kwds}
        # return named_dists.get(shape)(*params["args"], **params["kwds"])
        obj = cls(
            dist_family=shape, dist_params=None, empirical_data=None, skip_post=True
        )
        obj.dist = dist
        return obj

    # @classmethod
    # def dist_from_sps(
    #     cls, dist: sps.rv_continuous | sps.rv_discrete, shape: str = None
    # ):
    #     obj = cls.__new__(cls)  # bypass __init__ + __post_init__ logic
    #     obj.__dict__.update(
    #         dist_family=shape,
    #         dist_params=None,
    #         empirical_data=None,
    #         _dist=dist,
    #         _skip_post=True,
    #     )
    #     return obj

    # *  ---------------------conversion---------------------* #

    def to_pbox(self):
        """convert the distribution to a pbox
        note:
            - this only works for parameteried distributions for now
            - later on work with sample-approximated dist until `fit()`is implemented
        """
        if self._flag:
            params = self.dist_params
            if not isinstance(params, (tuple, list)):
                params = (params,)
            return named_pbox.get(self.dist_family)(*params)

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
    """Joint distribution class


    example:
        >>> from pyuncertainnumber import pba
        >>> dist_a, dist_b = pba.Distribution('gaussian', (5,1)), pba.Distribution('uniform', (2, 3))
        >>> c = pba.Dependency('gaussian', params=0.8)
        >>> joint_dist = pba.JointDistribution(copula=c, marginals=[dist_a, dist_b])
        >>> samples = joint_dist.sample(size=1000)
    """

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

    def __repr__(self):
        return f"a {self.ndim}-dimensional JointDistribution with copula: {self.copula.family} and marginals: {[m.dist_family for m in self.marginals]}"

    @staticmethod
    def from_sps(copula: Copula, marginals: list[sps.rv_continuous]):
        return CopulaDistribution(copula=copula, marginals=marginals)

    def sample(self, size, random_state=42):
        """generate orginal-space samples from the joint distribution"""
        return self._joint_dist.rvs(size, random_state=random_state)

    def u_sample(self, size, random_state=42):
        """generate copula-space samples from the joint distribution"""
        return self.copula.rvs(size, random_state=random_state)

    def joint_density_of_bi_grid(self, x: ArrayLike, y: ArrayLike):
        """compute the joint density on a grid given x and y arrays

        Used for bivariate arithmetic calculations of X and Y with designated (known) dependency and marginals.


        note:
            discretisation step sizes dx and dy are set up by the input x and y arrays

        example:
            >>> x = np.linspace(0, 1, 50)
            >>> y = np.linspace(0, 1, 50)
            >>> dep = Dependency("gaussian", params=0.7)
            >>> joint_density = dep.joint_density_of_grid(x, y)
        """
        XX, YY = np.meshgrid(x, y, indexing="ij")
        UU, VV = self.marginals[0].cdf(XX), self.marginals[1].cdf(
            YY
        )  # transform to unit square
        uv = np.column_stack([UU.ravel(), VV.ravel()])
        c_uv = self.copula.pdf(uv).reshape(XX.shape)  # copula density
        fX = self.marginals[0].pdf(XX)
        fY = self.marginals[1].pdf(YY)
        fXY = c_uv * fX * fY
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        return XX, YY, fXY, dx, dy

    @staticmethod
    def cdf_of_g(XX, YY, fXY, dx, dy, g_func, z_vals) -> ArrayLike:
        """Numerically approximate F_Z(z) = P(g(X,Y) <= z) via discretisation on a grid

        args:
            z_vals (ArrayLike): discretisation of z values (array) at which to compute F_Z
            XX, YY: the grid arrays from meshgrid
            fXY: joint density on the grid
            dx, dy: spacings in x and y directions
            g_func (callable): a general callable applied elementwise to (XX, YY)

        returns:
            FZ (ArrayLike): cumulative distribution function values at z_vals

        note:
            given precomputed grid (XX,YY), joint density fXY, and spacings.
        """
        G = g_func(XX, YY)  # evaluate g on the grid once
        FZ = np.empty_like(z_vals, dtype=float)
        for k, z in enumerate(z_vals):
            mask = G <= z
            FZ[k] = np.sum(fXY * mask) * dx * dy
        # Ensure numeric monotonicity (optional small fix)
        FZ = np.clip(np.maximum.accumulate(FZ), 0.0, 1.0)
        return FZ


# * --------------------- non-parametric ecdf cases --------------------- *#


class ECDF(Staircase):
    """Empirical cumulative distribution function (ecdf) class

    .. admonition:: Implementation

        supported by `Pbox` API hence samples will be degenerate intervals

    example:
        >>> import numpy as np
        >>> s = np.random.normal(size=1000)
        >>> ecdf = ECDF(s)
        >>> ecdf.plot()
    """

    def __init__(self, empirical_data: np.ndarray):
        left, p_values = get_ecdf(empirical_data)
        # TODO: quantile direct into Staircase. Hmm...
        super().__init__(left=left, right=left)


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


class LognormalSaneDist:
    def ppf(self, p_values, *params):
        dist = lognormal_sane(*params)
        return dist.ppf(p_values)

    def stats(self, *params, moments="mv"):
        dist = lognormal_sane(*params)
        return dist.stats(moments=moments)


def expon_sane(lamb):
    """Sane exponential distribution constructor"""
    return sps.expon(scale=1 / lamb)


class WrapperDist:
    def __init__(self, ppf_func):
        self._ppf = ppf_func

    def ppf(self, *args, **kwargs):
        return self._ppf(*args, **kwargs)


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
    "exponential": sps.expon,  # re-engineered exponential distribution
    "exponential_by_lambda": expon_sane,  # re-engineered exponential distribution
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
    "lognormal": LognormalSaneDist(),
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
    "weibull": sps.weibull_min,
    "weibull_min": sps.weibull_min,
    "weibull_max": sps.weibull_max,
    "wrapcauchy": sps.wrapcauchy,
    "bernoulli": sps.bernoulli,
    "betabinom": sps.betabinom,
    "binom": sps.binom,
    "binomial": sps.binom,
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
