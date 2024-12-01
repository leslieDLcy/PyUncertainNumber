"""distribution constructs """

import scipy.stats as sps
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


# * ------------------ sample-approximated dist representation  ------------------ *#
