"""
Deprecated as the all the codes herein are already moved to pbox.py
Kept for backward compatibility;
Construct a pbox from the specification of distributional parameters in the form of intervals.
In short, distribution-based p-boxes or parametric pboxes
"""

from PyUncertainNumber.characterisation.params import Params
from .interval import Interval
from .pbox_base import Pbox
import scipy.stats as sps
import numpy as np
import itertools
from .params import Params
from typing import *
from warnings import *


# TODO the __repr__ of a distribution is still showing as pbox, need to fix this

# a dict that links ''distribution name'' requiring specification to the scipy.stats distribution
dists = {
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
    "gaussian": sps.norm,
    "normal": sps.norm,
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


def __get_bounds(function_name=None, steps=Params.steps, *args):
    """ heavy lifting function that returns the bounds of the distribution for defining pboxes

    returns:
        Left: (array-like or list) left-bound of the pbox
        Right: (array-like or list) right-bound of the pbox
        mean: (Interval) mean of the pbox
        var: (Interval) variance of the pbox
    """

    # get bound arguments
    new_args = itertools.product(*args)

    bounds = []

    mean_hi = -np.inf
    mean_lo = np.inf
    var_lo = np.inf
    var_hi = 0

    for a in new_args:
        bounds.append(dists[function_name].ppf(Params.p_values, *a))
        bmean, bvar = dists[function_name].stats(*a, moments="mv")

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

    var = Interval(np.float64(var_lo), np.float64(var_hi))
    mean = Interval(np.float64(mean_lo), np.float64(mean_hi))

    Left = np.array(Left)
    Right = np.array(Right)

    return Left, Right, mean, var


# ---------------------supported distribution objects for pboxes ---------------------#

def lognormal(
    mean,
    var,
    steps=Params.steps,
):
    """
    Creates a p-box for the lognormal distribution

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
    ----------
    Pbox

    """
    if steps > 1000:
        x = np.linspace(1 / steps, 1 - 1 / steps, steps)
    else:
        x = np.linspace(0.001, 0.999, Params.steps)

    if mean.__class__.__name__ != "Interval":
        mean = Interval(mean, mean)
    if var.__class__.__name__ != "Interval":
        var = Interval(var, var)

    def __lognorm(mean, var):

        sigma = np.sqrt(np.log1p(var / mean**2))
        mu = np.log(mean) - 0.5 * sigma * sigma

        return sps.lognorm(sigma, loc=0, scale=np.exp(mu))

    bound0 = __lognorm(mean.left, var.left).ppf(x)
    bound1 = __lognorm(mean.right, var.left).ppf(x)
    bound2 = __lognorm(mean.left, var.right).ppf(x)
    bound3 = __lognorm(mean.right, var.right).ppf(x)

    Left = [min(bound0[i], bound1[i], bound2[i], bound3[i])
            for i in range(steps)]
    Right = [max(bound0[i], bound1[i], bound2[i], bound3[i])
             for i in range(steps)]

    Left = np.array(Left)
    Right = np.array(Right)
    return Pbox(Left, Right, steps=steps, shape="lognormal")


def alpha(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("alpha", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="alpha",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def anglit(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("anglit", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="anglit",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def arcsine(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("arcsine", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="arcsine",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def argus(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("argus", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="argus",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def beta(*args, steps=Params.steps):
    """
    Beta distribution
    """
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])
        if args[i].left == 0:
            args[i].left = 1e-5
        if args[i].right == 0:
            args[i].right = 1e-5

    Left, Right, mean, var = __get_bounds("beta", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="beta",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def betaprime(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("betaprime", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="betaprime",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def bradford(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("bradford", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="bradford",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def burr(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("burr", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="burr",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def burr12(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("burr12", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="burr12",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def cauchy(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("cauchy", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="cauchy",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def chi(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("chi", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="chi",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def chi2(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("chi2", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="chi2",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def cosine(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("cosine", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="cosine",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def crystalball(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("crystalball", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="crystalball",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def dgamma(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("dgamma", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="dgamma",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def dweibull(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("dweibull", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="dweibull",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def erlang(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("erlang", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="erlang",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def expon(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("expon", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="expon",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def exponnorm(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("exponnorm", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="exponnorm",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def exponweib(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("exponweib", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="exponweib",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def exponpow(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("exponpow", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="exponpow",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def f(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("f", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="f",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def fatiguelife(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("fatiguelife", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="fatiguelife",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def fisk(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("fisk", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="fisk",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def foldcauchy(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("foldcauchy", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="foldcauchy",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def foldnorm(mu, s, steps=Params.steps):

    x = np.linspace(0.0001, 0.9999, steps)
    if mu.__class__.__name__ != "Interval":
        mu = Interval(mu)
    if s.__class__.__name__ != "Interval":
        s = Interval(s)

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

    var = Interval(np.float64(var_lo), np.float64(var_hi))
    mean = Interval(np.float64(mean_lo), np.float64(mean_hi))

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


# def frechet_r(*args, steps = Params.steps):
#     args = list(args)
#     for i in range(0,len(args)):
#         if args[i].__class__.__name__ != 'Interval':
#             args[i] = Interval(args[i])

#     Left, Right, mean, var = __get_bounds('frechet_r',steps,*args)

#     return Pbox(
#           Left,
#           Right,
#           steps      = steps,
#           shape      = 'frechet_r',
#           mean_left  = mean.left,
#           mean_right = mean.right,
#           var_left   = var.left,
#           var_right  = var.right
#           )

# def frechet_l(*args, steps = Params.steps):
#     args = list(args)
#     for i in range(0,len(args)):
#         if args[i].__class__.__name__ != 'Interval':
#             args[i] = Interval(args[i])

#     Left, Right, mean, var = __get_bounds('frechet_l',steps,*args)

#     return Pbox(
#           Left,
#           Right,
#           steps      = steps,
#           shape      = 'frechet_l',
#           mean_left  = mean.left,
#           mean_right = mean.right,
#           var_left   = var.left,
#           var_right  = var.right
#           )


def genlogistic(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("genlogistic", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="genlogistic",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def gennorm(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("gennorm", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="gennorm",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def genpareto(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("genpareto", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="genpareto",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def genexpon(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("genexpon", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="genexpon",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def genextreme(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("genextreme", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="genextreme",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def gausshyper(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("gausshyper", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="gausshyper",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def gamma(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("gamma", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="gamma",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def gengamma(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("gengamma", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="gengamma",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def genhalflogistic(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("genhalflogistic", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="genhalflogistic",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def geninvgauss(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("geninvgauss", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="geninvgauss",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def gompertz(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("gompertz", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="gompertz",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def gumbel_r(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("gumbel_r", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="gumbel_r",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def gumbel_l(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("gumbel_l", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="gumbel_l",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def halfcauchy(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("halfcauchy", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="halfcauchy",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def halflogistic(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("halflogistic", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="halflogistic",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def halfnorm(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("halfnorm", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="halfnorm",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def halfgennorm(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("halfgennorm", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="halfgennorm",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def hypsecant(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("hypsecant", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="hypsecant",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def invgamma(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("invgamma", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="invgamma",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def invgauss(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("invgauss", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="invgauss",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def invweibull(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("invweibull", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="invweibull",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def johnsonsb(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("johnsonsb", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="johnsonsb",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def johnsonsu(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("johnsonsu", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="johnsonsu",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def kappa4(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("kappa4", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="kappa4",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def kappa3(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("kappa3", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="kappa3",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def ksone(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("ksone", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="ksone",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def kstwobign(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("kstwobign", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="kstwobign",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def laplace(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("laplace", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="laplace",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def levy(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("levy", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="levy",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def levy_l(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("levy_l", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="levy_l",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def levy_stable(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("levy_stable", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="levy_stable",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def logistic(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("logistic", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="logistic",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def loggamma(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("loggamma", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="loggamma",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def loglaplace(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("loglaplace", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="loglaplace",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def loguniform(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("loguniform", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="loguniform",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def lomax(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("lomax", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="lomax",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def maxwell(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("maxwell", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="maxwell",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def mielke(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("mielke", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="mielke",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def moyal(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("moyal", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="moyal",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def nakagami(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("nakagami", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="nakagami",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def ncx2(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("ncx2", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="ncx2",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def ncf(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("ncf", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="ncf",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def nct(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("nct", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="nct",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def norm(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("norm", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="norm",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def norminvgauss(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("norminvgauss", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="norminvgauss",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def pareto(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("pareto", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="pareto",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def pearson3(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("pearson3", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="pearson3",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def powerlaw(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("powerlaw", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="powerlaw",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def powerlognorm(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("powerlognorm", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="powerlognorm",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def powernorm(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("powernorm", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="powernorm",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def rdist(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("rdist", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="rdist",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def rayleigh(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("rayleigh", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="rayleigh",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def rice(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("rice", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="rice",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def recipinvgauss(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("recipinvgauss", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="recipinvgauss",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def semicircular(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("semicircular", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="semicircular",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def skewnorm(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("skewnorm", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="skewnorm",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def t(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("t", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="t",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def trapz(a, b, c, d, steps=Params.steps):
    if a.__class__.__name__ != "Interval":
        a = Interval(a)
    if b.__class__.__name__ != "Interval":
        b = Interval(b)
    if c.__class__.__name__ != "Interval":
        c = Interval(c)
    if d.__class__.__name__ != "Interval":
        d = Interval(d)

    x = np.linspace(0.0001, 0.9999, steps)
    left = sps.trapz.ppf(
        x, *sorted([b.lo() / d.lo(), c.lo() / d.lo(), a.lo(), d.lo() - a.lo()])
    )
    right = sps.trapz.ppf(
        x, *sorted([b.hi() / d.hi(), c.hi() / d.hi(), a.hi(), d.hi() - a.hi()])
    )

    return Pbox(left, right, steps=steps, shape="trapz")


def triang(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("triang", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="triang",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def truncexpon(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("truncexpon", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="truncexpon",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def truncnorm(left, right, mean=None, stddev=None, steps=Params.steps):

    if left.__class__.__name__ != "Interval":
        left = Interval(left)
    if right.__class__.__name__ != "Interval":
        right = Interval(right)
    if mean.__class__.__name__ != "Interval":
        mean = Interval(mean)
    if stddev.__class__.__name__ != "Interval":
        stddev = Interval(stddev)

    a, b = (left - mean) / stddev, (right - mean) / stddev

    Left, Right, mean, var = __get_bounds(
        "truncnorm", steps, a, b, mean, stddev)

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


def tukeylambda(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("tukeylambda", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="tukeylambda",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def uniform(a, b, steps=Params.steps):

    # x = np.linspace(0, 1, steps)
    # !SOLUTION: found a solution and prepare to raise an issue and submit a solution to Nick

    if a.__class__.__name__ != "Interval":
        a = Interval(a, a)
    if b.__class__.__name__ != "Interval":
        b = Interval(b, b)

    Left = np.linspace(a.left, b.left, steps)
    Right = np.linspace(a.right, b.right, steps)

    mean = 0.5 * (a + b)
    var = ((b - a) ** 2) / 12

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="uniform",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def vonmises(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("vonmises", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="vonmises",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def vonmises_line(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("vonmises_line", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="vonmises_line",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def wald(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("wald", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="wald",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def weibull_min(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("weibull_min", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="weibull_min",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def weibull_max(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("weibull_max", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="weibull_max",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def weibull(*args, steps=Params.steps):

    wm = weibull_max(*args)
    wl = weibull_min(*args)

    return Pbox(left=wl.left, right=wm.right)


def wrapcauchy(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("wrapcauchy", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="wrapcauchy",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def bernoulli(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("bernoulli", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="bernoulli",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def betabinom(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("betabinom", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="betabinom",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def binom(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("binom", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="binom",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def boltzmann(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("boltzmann", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="boltzmann",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def dlaplace(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("dlaplace", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="dlaplace",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def geom(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("geom", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="geom",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def hypergeom(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("hypergeom", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="hypergeom",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def logser(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("logser", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="logser",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def nbinom(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("nbinom", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="nbinom",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def planck(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("planck", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="planck",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def poisson(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("poisson", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="poisson",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def randint(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("randint", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="randint",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def skellam(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("skellam", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="skellam",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def zipf(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("zipf", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="zipf",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def yulesimon(*args, steps=Params.steps):
    args = list(args)
    for i in range(0, len(args)):
        if args[i].__class__.__name__ != "Interval":
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds("yulesimon", steps, *args)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="yulesimon",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


# Other distributions
def KM(k, m, steps=Params.steps):
    with catch_warnings():
        simplefilter("ignore")
        return beta(Interval(k, k + 1), Interval(m, m + 1), steps=steps)


def KN(k, n, steps=Params.steps):
    return KM(k, n - k, steps=steps)


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
    "lognormal": lognormal,
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


# ---------------------aliases---------------------#

# Alternate names
normal = norm
N = normal
unif = uniform
U = uniform
lognorm = lognormal


# def betapert(minimum, maximum, mode):
#     mu = (minimum + maximum + 4*mode)/6
#     alpha1 = (mu - minimum)*(2*mode - minimum - maximum)/((mode - mu)*(maximum - minimum))
#     alpha2 = alpha1*(maximum - mu)/(mu - minimum)
#     return minimum + (maximum - minimum) * beta(alpha1, alpha2)


# __all__ = [
#     "KM",
#     "KN",
#     "N",
#     "U",
#     "alpha",
#     "anglit",
#     "arcsine",
#     "argus",
#     "bernoulli",
#     "beta",
#     "betabinom",
#     # 'betapert',
#     "betaprime",
#     "binom",
#     "boltzmann",
#     "bradford",
#     "burr",
#     "burr12",
#     "cauchy",
#     "chi",
#     "chi2",
#     "cosine",
#     "crystalball",
#     "dgamma",
#     "dists",
#     "dlaplace",
#     "dweibull",
#     "erlang",
#     "expon",
#     "exponnorm",
#     "exponpow",
#     "exponweib",
#     "f",
#     "fatiguelife",
#     "fisk",
#     "foldcauchy",
#     "foldnorm",
#     "gamma",
#     "gausshyper",
#     "genexpon",
#     "genextreme",
#     "gengamma",
#     "genhalflogistic",
#     "geninvgauss",
#     "genlogistic",
#     "gennorm",
#     "genpareto",
#     "geom",
#     # 'gibrat',
#     "gompertz",
#     "gumbel_l",
#     "gumbel_r",
#     "halfcauchy",
#     "halfgennorm",
#     "halflogistic",
#     "halfnorm",
#     "hypergeom",
#     "hypsecant",
#     "invgamma",
#     "invgauss",
#     "invweibull",
#     "itertools",
#     "johnsonsb",
#     "johnsonsu",
#     "kappa3",
#     "kappa4",
#     "ksone",
#     "kstwobign",
#     "laplace",
#     "levy",
#     "levy_l",
#     "levy_stable",
#     "loggamma",
#     "logistic",
#     "loglaplace",
#     "lognorm",
#     "lognormal",
#     "logser",
#     "loguniform",
#     "lomax",
#     "maxwell",
#     "mielke",
#     "moyal",
#     "nakagami",
#     "nbinom",
#     "ncf",
#     "nct",
#     "ncx2",
#     "norm",
#     "normal",
#     "norminvgauss",
#     "np",
#     "pareto",
#     "pearson3",
#     "planck",
#     "poisson",
#     "powerlaw",
#     "powerlognorm",
#     "powernorm",
#     "randint",
#     "rayleigh",
#     "rdist",
#     "recipinvgauss",
#     "rice",
#     "semicircular",
#     "skellam",
#     "skewnorm",
#     "sps",
#     "t",
#     "trapz",
#     "triang",
#     "truncexpon",
#     "truncnorm",
#     "tukeylambda",
#     "unif",
#     "uniform",
#     "vonmises",
#     "vonmises_line",
#     "wald",
#     "weibull",
#     "weibull_max",
#     "weibull_min",
#     "wrapcauchy",
#     "yulesimon",
#     "zipf",
# ]
