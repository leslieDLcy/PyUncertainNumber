""" contains codes for parametric p-boxes, which seems contradictory to the `dists`
file and seems to be deprecated. """
"""

from .pbox_base import Pbox
from .interval import Interval
from .params import Params

import scipy.stats as sps
import numpy as np
import itertools
import sys

# the dist loop up table
dist = [
    "alpha",
    "anglit",
    "arcsine",
    "argus",
    "beta",
    "betaprime",
    "bradford",
    "burr",
    "burr12",
    "cauchy",
    "chi",
    "chi2",
    "cosine",
    "crystalball",
    "dgamma",
    "dweibull",
    "erlang",
    "expon",
    "exponnorm",
    "exponweib",
    "exponpow",
    "f",
    "fatiguelife",
    "fisk",
    "foldcauchy",
    "foldnorm",
    "genlogistic",
    "gennorm",
    "genpareto",
    "genexpon",
    "genextreme",
    "gausshyper",
    "gamma",
    "gengamma",
    "genhalflogistic",
    "geninvgauss",
    "gilbrat",
    "gompertz",
    "gumbel_r",
    "gumbel_l",
    "halfcauchy",
    "halflogistic",
    "halfnorm",
    "halfgennorm",
    "hypsecant",
    "invgamma",
    "invgauss",
    "invweibull",
    "johnsonsb",
    "johnsonsu",
    "kappa4",
    "kappa3",
    "ksone",
    "kstwobign",
    "laplace",
    "levy",
    "levy_l",
    "levy_stable",
    "logistic",
    "loggamma",
    "loglaplace",
    "lognorm",
    "loguniform",
    "lomax",
    "maxwell",
    "mielke",
    "moyal",
    "nakagami",
    "ncx2",
    "ncf",
    "nct",
    "norm",
    "norminvgauss",
    "pareto",
    "pearson3",
    "powerlaw",
    "powerlognorm",
    "powernorm",
    "rdist",
    "rayleigh",
    "rice",
    "recipinvgauss",
    "semicircular",
    "skewnorm",
    "t",
    "trapz",
    "triang",
    "truncexpon",
    "truncnorm",
    "tukeylambda",
    "uniform",
    "vonmises",
    "vonmises_line",
    "wald",
    "weibull_min",
    "weibull_max",
    "wrapcauchy",
    "bernoulli",
    "betabinom",
    "binom",
    "boltzmann",
    "dlaplace",
    "geom",
    "hypergeom",
    "logser",
    "nbinom",
    "planck",
    "poisson",
    "randint",
    "skellam",
    "zipf",
    "yulesimon",
]

__all__ = ["Parametric"] + dist


extra = {
    "lognorm": sps.lognorm,
    "foldnorm": sps.foldnorm,
    "trapz": sps.trapz,
    "truncnorm": sps.truncnorm,
    "uniform": sps.uniform,
    "beta": sps.beta,
}


class Bounds:
    """ the template class for parametric p-boxes """

    STEPS = Params.steps

    def __init__(self, shape, *args, n_subinterval=5):
        self.shape = shape
        self.bounds = get_distributions(self.shape, *args, n_subinterval=n_subinterval)
        self.pbox = self._pba_constructor(*args)

    def _pba_constructor(self, *args):
        # args = list(args)

        # for i in range(len(args)):
        #     if args[i].__class__.__name__ != 'Interval':
        #         args[i] = Interval(args[i])

        Left, Right, mean, var = get_bounds(self.shape, self.STEPS, *args)
        return Pbox(
            Left,
            Right,
            steps=self.STEPS,
            shape=self.shape,
            mean_left=mean.left,
            mean_right=mean.right,
            var_left=var.left,
            var_right=var.right,
        )


class Parametric(Bounds):
    """
    A parametric Pbox is defined where parameters of a named distribtuion are specified as
    Intervals. This class wraps the `scipy.stats` library and supports all scipy methods such as
    pdf, cdf, survival function etc.

    Parametric can be created using any combination of the following styles:norm
        pba.Parametric('', [0,1], [1,2])
        pba.Parametric('cauchy', Interval[0,1], 1)
        pba.Parametric('beta', a = Interval[0,.5], b=0.5)


    Parameters
    ----------
    shape : str
        name of the distribution
    **args : float, list, np.array, Interval
        set of distribution parameters
    **kwargs :
        set of key value pairs for the distribution parameters

    Attributes
    ----------
    left : numeric
        left side of interval
    right : numeric
        right side of interval

    """

    params = []
    __pbox__ = True

    def __init__(self, shape, *args, n_subinterval=5, **kwargs):
        self.params = list_parameters(shape)
        self.shape = shape

        if args:
            args = args2int(*args)
            self.set_from_args(*args)
        if kwargs:
            kwargs = self._set_support(**kwargs)
            self.set_parameters(**kwargs)
            args = [v for i, v in kwargs.items()]

        super().__init__(self.shape, *args, n_subinterval=n_subinterval)

    def __retter__(self):
        return self

    def _set_support(self, **kwargs):
        v = [v for k, v in kwargs.items() if k == "support"]
        if v:
            del kwargs["support"]
        self.scale_support = v[0]
        return kwargs

    def get_parameter_values(self):
        return [getattr(self, k) for k in self.params]

    def set_from_args(self, *args):
        self.params
        args = list(args)
        for i, v in enumerate(args):
            # if not isinstance(v, Interval):
            #     v = Interval(v)
            d = {self.params[i]: v}
            self._set_parameter(**d)

    def set_parameters(self, **kwargs):
        if kwargs:
            self._set_parameter(**kwargs)

    def _set_parameter(self, **kwargs):
        if kwargs:
            for k, v in kwargs.items():
                assert k in self.params, "{} not in param list: {}".format(
                    k, self.params
                )
                if not isinstance(v, Interval):
                    v = Interval(v)
                setattr(self, k, v)

    def get_support(self):
        if hasattr(self, "scale_support"):
            print("Support Scaled")
            return self.scale_support
        else:
            return self.__getattr__("support")()

    def _get_distributions_method(self, name):
        m = {}
        for k, v in self.bounds.items():
            m[k] = getattr(v["dist"], name)
        return m

    def __getattr__(self, name):
        sample_methods = [
            "cdf",
            "logcdf",
            "logpdf",
            "isf",
            "logpmf",
            "logsf",
            "sf",
            "pdf",
            "pmf",
            "ppf",
        ]

        dist_methods = [
            "entropy",
            "expect",
            "interval",
            "isf",
            "mean",
            "median",
            "moment",
            "random_state",
            "rvs",
            "stats",
            "std",
            "support",
            "var",
        ]
        try:
            if name in sample_methods:
                m = self._get_distributions_method(name)
                if hasattr(self, "scale_support"):  # self.shape == 'beta':

                    def F(*x):
                        return run_fun_scaled(m, *x, user_support=self.scale_support)

                else:

                    def F(*x):
                        return run_fun(m, *x)

                return F

            elif name in dist_methods:
                m = self._get_distributions_method(name)

                def F(*n):
                    return dist_fun(m, *n)

                return F
            else:
                return getattr(self.pbox, name)
        except AttributeError:
            raise AttributeError("Bounds' object has no attribute '%s'" % name)

    # def __getattr__(self, name):
    #     try:
    #         return getattr(self.pbox, name)
    #     except:
    #         try:
    #             return getattr(self.bounds, name)
    #         except AttributeError:
    #             raise AttributeError("Parametric' object has no attribute '%s'" % name)


def Z_conv(x, mi=0, ma=1):
    return (x - mi) / (ma - mi)


def X_conv(z, mi=0, ma=1):
    return (z + mi) * (ma - mi)


def args2int(*args):
    args = list(args)
    for i, a in enumerate(args):
        if not isinstance(a, Interval):
            args[i] = Interval(a)
    return args


def check_implimentation(distribution):
    if distribution in dist:
        return True
    else:
        sys.exit(
            "{} not implimented, choose form : \n\n{}".format(
                distribution, availiable_distributions()
            )
        )
        return False


def availiable_distributions():
    return [" {},".format(d) for d in dist]


def list_parameters(distribution):
    """List parameters for stats.distribution.

    Parameters
    ----------
        distribution: a string or stats distribution object.
    Output
    ------
        A list of distribution parameter strings.
    """
    if isinstance(distribution, str):
        try:
            distribution = getattr(sps, distribution)
        except:
            check_implimentation(distribution)
    if distribution.shapes:
        parameters = [name.strip() for name in distribution.shapes.split(",")]
    else:
        parameters = []
    if distribution.name in sps._discrete_distns._distn_names:
        parameters.insert(0, "loc")
    elif distribution.name in sps._continuous_distns._distn_names:
        parameters.insert(0, "loc")
        parameters.insert(1, "scale")
    return parameters


def get_distributions(distribution, *args, n_subinterval=5):

    args = list(args)
    if n_subinterval:
        args = [subintervalise(i, n_subinterval) for i in args]
    new_args = itertools.product(*args)

    bounds = {}
    i = 0
    for a in new_args:
        bounds[i] = {}
        bounds[i]["dist"] = getattr(sps, distribution)(*a)
        bounds[i]["param"] = a
        i += 1
    return bounds


def get_bounds(distribution, support=[1e-5, 1 - 1e-5], *args):
    # define support
    steps = 200
    x = np.linspace(1e-5, 1 - 1e-5, 200)
    # get bound arguments
    new_args = itertools.product(*args)

    bounds = []
    mean_hi = -np.inf
    mean_lo = np.inf
    var_lo = np.inf
    var_hi = 0

    for a in new_args:
        bounds.append(getattr(sps, distribution).ppf(x, *a))
        bmean, bvar = getattr(sps, distribution).stats(*a, moments="mv")
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


def subintervalise(interval, n):
    xi = np.linspace(interval.left, interval.right, n)
    x = np.hstack([xi[:-1], xi[1:]])
    return x.T


def run_fun(m, *x):
    if x:
        x = x[0]
        l = [g(x) for j, g in m.items()]
    else:
        l = [g() for j, g in m.items()]
        mi, ma = np.min(l, axis=0), np.max(l, axis=0)
        # I = zip(mi, ma)
        # return [Interval(i) for i in I]
        return Interval(min(mi), max(ma))

    if isinstance(x, float) or isinstance(x, int):
        return Interval(min(l), max(l))

    if isinstance(x, list) or isinstance(x, np.ndarray):
        l = np.array(l)
        mi, ma = np.min(l, axis=0), np.max(l, axis=0)
        I = zip(mi, ma)
        return [Interval(i) for i in I]


def run_fun_scaled(m, *x, dist_support=[0, 1], user_support=[0, 1]):
    if x:
        x = x[0]
        xi = Z_conv(x, mi=user_support[0], ma=user_support[1])
        l = [g(xi) for j, g in m.items()]
    else:
        l = [g() for j, g in m.items()]
        mi, ma = np.min(l, axis=0), np.max(l, axis=0)
        I = zip(mi, ma)
        return [Interval(i) for i in I]

    if isinstance(xi, float) or isinstance(x, int):
        return Interval(min(l), max(l))

    if isinstance(xi, list) or isinstance(xi, np.ndarray):
        l = np.array(l)
        mi, ma = np.min(l, axis=0), np.max(l, axis=0)
        I = zip(mi, ma)
        return [Interval(i) for i in I]


def dist_fun(m, *n):
    # TODO: This method can access rvs, not clear what the behaviour should be though.
    # Figure out what a random sample of the pbox should be and do that.
    # I think probably Intervals, so should use inverse transfrom. Maybe should overwrite RVS
    if n:
        n = n[0]
        l = [g(n) for j, g in m.items()]

    else:
        l = [g() for j, g in m.items()]

    mi, ma = np.min(l, axis=0), np.max(l, axis=0)
    if isinstance(mi, float):
        return Interval(mi, ma)
    else:
        I = zip(mi, ma)
        return [Interval(i) for i in I]


# class Normal(Parametric):
#     def __init__(self,*args, **kwargs):
#         super().__init__('norm', *args, **kwargs)

# class t(Parametric):
#     def __init__(self,*args, **kwargs):
#         super().__init__('t', *args, **kwargs)


def trapz(a, b, c, d, steps=200):
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


dl = {}
for funcname in dist:

    def func(*args, name=funcname, **kwargs):
        f"""
        Generate parametric p-box for {name}
        """
        return Parametric(name, *args, **kwargs).__retter__()

    # dl[funcname] = func
    setattr(sys.modules[__name__], funcname, func)



