from __future__ import annotations
import functools

from .pbox_abc import Pbox, Leaf, pbox_from_extredists
import scipy.stats as sps
import numpy as np
import itertools
from .params import Params
from typing import *
from warnings import *
from .intervals.intervalOperators import wc_scalar_interval

if TYPE_CHECKING:
    from .pbox_abc import Pbox


""" parametric pboxes"""


def makePbox(func) -> Pbox:
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        family_str = func(*args, **kwargs)
        return _bound_pcdf(family_str, *args)

    return wrapper_decorator


# top-level
def _bound_pcdf(dist_family, *args, **kwargs):
    """bound the parametric CDF

    note:
        - top-level implemenatation
        - only support fully bounded parameters
    """
    from .pbox_abc import Leaf

    Left, Right, mean, var = _parametric_bounds(dist_family, *args, **kwargs)
    return Leaf(
        left=Left, right=Right, shape=dist_family, dist_params=args, mean=mean, var=var
    )


def _parametric_bounds(dist_family, *args, steps=Params.steps):
    """from parametric distribution specification to define the lower and upper bound of the p-box

    args:
        - dist_family: (str) the name of the distribution
        - *args : several parameter (interval or list)

    note:
        - middle level implementation

    """

    from .distributions import named_dists
    from .intervals.number import Interval as I

    # TODO logically speaking, it can be (0,1) ergo support will be [-inf, inf] see it works with other part codes
    # define percentile range thus getting the support

    i_args = [wc_scalar_interval(b) for b in args]

    p = Params.p_values

    # get bound arguments
    new_args = itertools.product(*[i.to_numpy() for i in i_args])

    bounds = []

    mean_hi = -np.inf
    mean_lo = np.inf
    var_lo = np.inf
    var_hi = 0

    for a in new_args:
        bounds.append(named_dists[dist_family].ppf(p, *a))  # bounds quantiles
        bmean, bvar = named_dists[dist_family].stats(*a, moments="mv")

        if bmean < mean_lo:
            mean_lo = bmean
        if bmean > mean_hi:
            mean_hi = bmean
        if bvar > var_hi:
            var_hi = bvar
        if bvar < var_lo:
            var_lo = bvar

    Left = np.array([min([b[i] for b in bounds]) for i in range(steps)])
    Right = np.array([max([b[i] for b in bounds]) for i in range(steps)])

    var = I(var_lo, var_hi)
    mean = I(mean_lo, mean_hi)

    return Left, Right, mean, var


# * ---------------------supported distributional pboxes ---------------------#


@makePbox
def norm(*args):
    return "norm"


@makePbox
def alpha(*args):
    return "alpha"


@makePbox
def anglit(*args):
    return "anglit"


@makePbox
def argus(*args):
    return "argus"


@makePbox
def arcsine(*args):
    return "arcsine"


@makePbox
def beta(*args):
    return "beta"


@makePbox
def betaprime(*args):
    return "betaprime"


@makePbox
def bradford(*args):
    return "bradford"


@makePbox
def burr(*args):
    return "burr"


@makePbox
def burr12(*args):
    return "burr12"


@makePbox
def cauchy(*args):
    return "cauchy"


@makePbox
def chi(*args):
    return "chi"


@makePbox
def chi2(*args):
    return "chi2"


@makePbox
def cosine(*args):
    return "cosine"


@makePbox
def crystalball(*args):
    return "crystalball"


@makePbox
def dgamma(*args):
    return "dgamma"


@makePbox
def dweibull(*args):
    return "dweibull"


@makePbox
def erlang(*args):
    return "erlang"


# special case due to kwargs (scale)
# @makePbox
# def expon(*args):
#     return "expon"


@makePbox
def exponnorm(*args):
    return "exponnorm"


@makePbox
def exponweib(*args):
    return "exponweib"


@makePbox
def exponpow(*args):
    return "exponpow"


@makePbox
def f(*args):
    return "f"


@makePbox
def fatiguelife(*args):
    return "fatiguelife"


@makePbox
def fisk(*args):
    return "fisk"


@makePbox
def foldcauchy(*args):
    return "foldcauchy"


def foldnorm(mu, s, steps=Params.steps):

    x = np.linspace(0.0001, 0.9999, steps)
    if mu.__class__.__name__ != "wc_scalar_interval":
        mu = wc_scalar_interval(mu)
    if s.__class__.__name__ != "wc_scalar_interval":
        s = wc_scalar_interval(s)

    new_args = [
        [mu.lo() / s.lo(), 0, s.lo()],
        [mu.hi() / s.lo(), 0, s.lo()],
        [mu.lo() / s.hi(), 0, s.hi()],
        [mu.hi() / s.hi(), 0, s.hi()],
    ]

    bounds = []

    mean_hi = -np.inf
    mean_lo = np.inf
    var_lo = np.inf
    var_hi = 0

    for a in new_args:

        bounds.append(sps.foldnorm.ppf(x, *a))
        bmean, bvar = sps.foldnorm.stats(*a, moments="mv")

        if bmean < mean_lo:
            mean_lo = bmean
        if bmean > mean_hi:
            mean_hi = bmean
        if bvar > var_hi:
            var_hi = bvar
        if bvar < var_lo:
            var_lo = bvar

    Left = [min([b[i] for b in bounds]) for i in range(steps)]
    Right = [max([b[i] for b in bounds]) for i in range(steps)]

    var = wc_scalar_interval(np.float64(var_lo), np.float64(var_hi))
    mean = wc_scalar_interval(np.float64(mean_lo), np.float64(mean_hi))

    Left = np.array(Left)
    Right = np.array(Right)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="foldnorm",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


@makePbox
def genlogistic(*args):
    return "genlogistic"


@makePbox
def gennorm(*args):
    return "gennorm"


@makePbox
def genpareto(*args):
    return "genpareto"


@makePbox
def genexpon(*args):
    return "genexpon"


@makePbox
def genextreme(*args):
    return "genextreme"


@makePbox
def gausshyper(*args):
    return "gausshyper"


@makePbox
def gamma(*args):
    return "gamma"


@makePbox
def gengamma(*args):
    return "gengamma"


@makePbox
def genhalflogistic(*args):
    return "genhalflogistic"


@makePbox
def geninvgauss(*args):
    return "geninvgauss"


@makePbox
def gompertz(*args):
    return "gompertz"


@makePbox
def gumbel_r(*args):
    return "gumbel_r"


@makePbox
def gumbel_l(*args):
    return "gumbel_l"


@makePbox
def halfcauchy(*args):
    return "halfcauchy"


@makePbox
def halflogistic(*args):
    return "halflogistic"


@makePbox
def halfnorm(*args):
    return "halfnorm"


@makePbox
def halfgennorm(*args):
    return "halfgennorm"


@makePbox
def hypsecant(*args):
    return "hypsecant"


@makePbox
def invgamma(*args):
    return "invgamma"


@makePbox
def invgauss(*args):
    return "invgauss"


@makePbox
def invweibull(*args):
    return "invweibull"


@makePbox
def irwinhall(*args):
    return "irwinhall"


@makePbox
def jf_skew_t(*args):
    return "jf_skew_t"


@makePbox
def johnsonsb(*args):
    return "johnsonsb"


@makePbox
def johnsonsu(*args):
    return "johnsonsu"


@makePbox
def kappa4(*args):
    return "kappa4"


@makePbox
def kappa3(*args):
    return "kappa3"


@makePbox
def ksone(*args):
    return "ksone"


@makePbox
def kstwo(*args):
    return "kstwo"


@makePbox
def kstwobign(*args):
    return "kstwobign"


@makePbox
def laplace(*args):

    return "laplace"


@makePbox
def laplace_asymmetric(*args):
    return "laplace_asymmetric"


@makePbox
def levy(*args):
    return "levy"


@makePbox
def levy_l(*args):
    return "levy_l"


@makePbox
def levy_stable(*args):
    return "levy_stable"


@makePbox
def logistic(*args):
    return "logistic"


@makePbox
def loggamma(*args):
    return "loggamma"


@makePbox
def loglaplace(*args):
    return "loglaplace"


@makePbox
def loguniform(*args):
    return "loguniform"


@makePbox
def lomax(*args):
    return "lomax"


@makePbox
def maxwell(*args):
    return "maxwell"


@makePbox
def mielke(*args):
    return "mielke"


@makePbox
def moyal(*args):
    return "moyal"


@makePbox
def nakagami(*args):
    return "nakagami"


@makePbox
def ncx2(*args):
    return "ncx2"


@makePbox
def ncf(*args):
    return "ncf"


@makePbox
def nct(*args):
    return "nct"


@makePbox
def norminvgauss(*args):
    return "norminvgauss"


@makePbox
def pareto(*args):
    return "pareto"


@makePbox
def pearson3(*args):
    return "pearson3"


@makePbox
def powerlaw(*args):
    return "powerlaw"


@makePbox
def powerlognorm(*args):
    return "powerlognorm"


@makePbox
def powernorm(*args):
    return "powernorm"


@makePbox
def rdist(*args):
    return "rdist"


@makePbox
def rayleigh(*args):
    return "rayleigh"


@makePbox
def rel_breitwigner(*args):
    return "rel_breitwigner"


@makePbox
def rice(*args):
    return "rice"


@makePbox
def recipinvgauss(*args):
    return "recipinvgauss"


@makePbox
def semicircular(*args):
    return "semicircular"


@makePbox
def skewcauchy(*args):
    return "skewcauchy"


@makePbox
def skewnorm(*args):
    return "skewnorm"


@makePbox
def studentized_range(*args):
    return "studentized_range"


@makePbox
def t(*args):
    return "t"


@makePbox
def trapezoid(*args):
    return "trapezoid"


@makePbox
def triang(*args):
    return "triang"


@makePbox
def truncexpon(*args):
    return "truncexpon"


@makePbox
def truncnorm(*args):
    return "truncnorm"


@makePbox
def truncpareto(*args):
    return "truncpareto"


@makePbox
def truncweibull_min(*args):
    return "truncweibull_min"


@makePbox
def tukeylambda(*args):
    return "tukeylambda"


def uniform_sps(*args):
    return "uniform"


@makePbox
def vonmises(*args):
    return "vonmises"


@makePbox
def vonmises_line(*args):
    return "vonmises_line"


@makePbox
def wald(*args):
    return "wald"


@makePbox
def weibull_min(*args):
    return "weibull_min"


@makePbox
def weibull_max(*args):
    return "weibull_max"


@makePbox
def wrapcauchy(*args):
    return "wrapcauchy"


# *---------------------some special ones ---------------------*#


def lognormal_weird(
    mean,
    var,
    steps=Params.steps,
):
    """p-box for the lognormal distribution

    *Note: the parameters used are the mean and variance of the lognormal distribution

    not the mean and variance of the underlying normal*
    See:
    `[1]<https://en.wikipedia.org/wiki/Log-normal_distribution#Generation_and_parameters>`
    `[2]<https://stackoverflow.com/questions/51906063/distribution-mean-and-standard-deviation-using-scipy-stats>`


    Parameters
    ----------
    mean :
        mean of the lognormal distribution
    var :
        variance of the lognormal distribution

    Returns
    -------
    Pbox

    """

    x = np.linspace(0.001, 0.999, Params.steps)
    mean, var = wc_scalar_interval(mean), wc_scalar_interval(var)

    def __lognorm(mean, var):
        sigma = np.sqrt(np.log1p(var / mean**2))
        mu = np.log(mean) - 0.5 * sigma * sigma
        return sps.lognorm(sigma, loc=0, scale=np.exp(mu))

    bound0 = __lognorm(mean.left, var.left).ppf(x)
    bound1 = __lognorm(mean.right, var.left).ppf(x)
    bound2 = __lognorm(mean.left, var.right).ppf(x)
    bound3 = __lognorm(mean.right, var.right).ppf(x)

    Left = [min(bound0[i], bound1[i], bound2[i], bound3[i]) for i in range(steps)]
    Right = [max(bound0[i], bound1[i], bound2[i], bound3[i]) for i in range(steps)]

    Left = np.array(Left)
    Right = np.array(Right)
    return Leaf(left=Left, right=Right, steps=steps, shape="lognormal")


# @makePbox
# def lognormal(*args):
#     return "lognormal"


def uniform(a, b, steps=Params.steps):
    """special case of Uniform distribution as
    Scipy has an unbelivably strange parameterisation than common sense

    args:
        - a: (float) lower endpoint
        - b: (float) upper endpoints
    """

    # loc, scale = uniform_reparameterisation(a,  b)
    # return uniform_sps(loc, scale)

    a, b = [wc_scalar_interval(arg) for arg in [a, b]]

    Left = np.linspace(a.left, b.left, steps)
    Right = np.linspace(a.right, b.right, steps)

    return Leaf(
        left=Left,
        right=Right,
        steps=steps,
        shape="uniform",
    )


def expon(scale, loc=0, steps=Params.steps):
    return _bound_pcdf("expon", loc, scale)


def trapz(a, b, c, d, steps=Params.steps):
    if a.__class__.__name__ != "wc_scalar_interval":
        a = wc_scalar_interval(a)
    if b.__class__.__name__ != "wc_scalar_interval":
        b = wc_scalar_interval(b)
    if c.__class__.__name__ != "wc_scalar_interval":
        c = wc_scalar_interval(c)
    if d.__class__.__name__ != "wc_scalar_interval":
        d = wc_scalar_interval(d)

    x = np.linspace(0.0001, 0.9999, steps)
    left = sps.trapz.ppf(
        x, *sorted([b.lo() / d.lo(), c.lo() / d.lo(), a.lo(), d.lo() - a.lo()])
    )
    right = sps.trapz.ppf(
        x, *sorted([b.hi() / d.hi(), c.hi() / d.hi(), a.hi(), d.hi() - a.hi()])
    )

    return Pbox(left, right, steps=steps, shape="trapz")


def truncnorm(left, right, mean=None, stddev=None, steps=Params.steps):

    if left.__class__.__name__ != "wc_scalar_interval":
        left = wc_scalar_interval(left)
    if right.__class__.__name__ != "wc_scalar_interval":
        right = wc_scalar_interval(right)
    if mean.__class__.__name__ != "wc_scalar_interval":
        mean = wc_scalar_interval(mean)
    if stddev.__class__.__name__ != "wc_scalar_interval":
        stddev = wc_scalar_interval(stddev)

    a, b = (left - mean) / stddev, (right - mean) / stddev

    Left, Right, mean, var = _parametric_bounds("truncnorm", steps, a, b, mean, stddev)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="truncnorm",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def weibull(*args, steps=Params.steps):

    wm = weibull_max(*args)
    wl = weibull_min(*args)

    return Pbox(left=wl.left, right=wm.right)


# Other distributions
def KM(k, m, steps=Params.steps):
    with catch_warnings():
        simplefilter("ignore")
        return beta(
            wc_scalar_interval(k, k + 1), wc_scalar_interval(m, m + 1), steps=steps
        )


def KN(k, n, steps=Params.steps):
    return KM(k, n - k, steps=steps)


# *---------------------discrete distributions---------------------*#


@makePbox
def bernoulli(*args):
    return "bernoulli"


@makePbox
def betabinom(*args):
    return "betabinom"


@makePbox
def betanbinom(*args):
    return "betanbinom"


@makePbox
def binom(*args):
    return "binom"


@makePbox
def boltzmann(*args):
    return "boltzmann"


@makePbox
def dlaplace(*args):
    return "dlaplace"


@makePbox
def geom(*args):
    return "geom"


@makePbox
def hypergeom(*args):
    return "hypergeom"


@makePbox
def logser(*args):
    return "logser"


@makePbox
def nbinom(*args):
    return "nbinom"


@makePbox
def nchypergeom_fisher(*args):
    return "nchypergeom_fisher"


@makePbox
def nchypergeom_wallenius(*args):
    return "nchypergeom_wallenius"


@makePbox
def nhypergeom(*args):
    return "nhypergeom"


@makePbox
def planck(*args):
    return "planck"


@makePbox
def poisson(*args):
    return "poisson"


@makePbox
def randint(*args):
    return "randint"


@makePbox
def skellam(*args):
    return "skellam"


@makePbox
def yulesimon(*args):
    return "yulesimon"


@makePbox
def zipf(*args):
    return "zipf"


@makePbox
def zipfian(*args):
    return "zipfian"


# *---------------------aliases---------------------*#
normal = norm
gaussian = norm
lognormal = lognormal_weird

# *---------------------named pboxes for UN ---------------------*#
named_pbox = {
    "alpha": alpha,
    "anglit": anglit,
    "arcsine": arcsine,
    "argus": argus,
    "beta": beta,
    "betaprime": betaprime,
    "bradford": bradford,
    "burr": burr,
    "burr12": burr12,
    "cauchy": cauchy,
    "chi": chi,
    "chi2": chi2,
    "cosine": cosine,
    "crystalball": crystalball,
    "dgamma": dgamma,
    "dweibull": dweibull,
    "erlang": erlang,
    "expon": expon,
    "exponnorm": exponnorm,
    "exponweib": exponweib,
    "exponpow": exponpow,
    "f": f,
    "fatiguelife": fatiguelife,
    "fisk": fisk,
    "foldcauchy": foldcauchy,
    "foldnorm": foldnorm,
    # 'frechet_r' : frechet_r,
    # 'frechet_l' : frechet_l,
    "genlogistic": genlogistic,
    "gennorm": gennorm,
    "genpareto": genpareto,
    "genexpon": genexpon,
    "genextreme": genextreme,
    "gausshyper": gausshyper,
    "gamma": gamma,
    "gengamma": gengamma,
    "genhalflogistic": genhalflogistic,
    "geninvgauss": geninvgauss,
    # 'gibrat' : gibrat,
    "gompertz": gompertz,
    "gumbel_r": gumbel_r,
    "gumbel_l": gumbel_l,
    "halfcauchy": halfcauchy,
    "halflogistic": halflogistic,
    "halfnorm": halfnorm,
    "halfgennorm": halfgennorm,
    "hypsecant": hypsecant,
    "invgamma": invgamma,
    "invgauss": invgauss,
    "invweibull": invweibull,
    "johnsonsb": johnsonsb,
    "johnsonsu": johnsonsu,
    "kappa4": kappa4,
    "kappa3": kappa3,
    "ksone": ksone,
    "kstwobign": kstwobign,
    "laplace": laplace,
    "levy": levy,
    "levy_l": levy_l,
    "levy_stable": levy_stable,
    "logistic": logistic,
    "loggamma": loggamma,
    "loglaplace": loglaplace,
    "lognormal": lognormal_weird,
    "loguniform": loguniform,
    "lomax": lomax,
    "maxwell": maxwell,
    "mielke": mielke,
    "moyal": moyal,
    "nakagami": nakagami,
    "ncx2": ncx2,
    "ncf": ncf,
    "nct": nct,
    "norm": norm,
    "normal": norm,
    "gaussian": norm,
    "norminvgauss": norminvgauss,
    "pareto": pareto,
    "pearson3": pearson3,
    "powerlaw": powerlaw,
    "powerlognorm": powerlognorm,
    "powernorm": powernorm,
    "rdist": rdist,
    "rayleigh": rayleigh,
    "rice": rice,
    "recipinvgauss": recipinvgauss,
    "semicircular": semicircular,
    "skewnorm": skewnorm,
    "t": t,
    "trapz": trapz,
    "triang": triang,
    "truncexpon": truncexpon,
    "truncnorm": truncnorm,
    "tukeylambda": tukeylambda,
    "uniform": uniform,
    "vonmises": vonmises,
    "vonmises_line": vonmises_line,
    "wald": wald,
    "weibull_min": weibull_min,
    "weibull_max": weibull_max,
    "wrapcauchy": wrapcauchy,
    "bernoulli": bernoulli,
    "betabinom": betabinom,
    "binom": binom,
    "boltzmann": boltzmann,
    "dlaplace": dlaplace,
    "geom": geom,
    "hypergeom": hypergeom,
    "logser": logser,
    "nbinom": nbinom,
    "planck": planck,
    "poisson": poisson,
    "randint": randint,
    "skellam": skellam,
    "zipf": zipf,
    "yulesimon": yulesimon,
}
