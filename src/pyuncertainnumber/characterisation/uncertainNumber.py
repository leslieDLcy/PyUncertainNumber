from __future__ import annotations
from typing import TYPE_CHECKING
import functools

from pyuncertainnumber.pba.operation import convert
from .un_fields import *
from .utils import *
from .config import Config
from pathlib import Path
from ..nlp.language_parsing import hedge_interpret
from .check import DistributionSpecification
from ..pba.pbox_parametric import named_pbox
from ..pba.intervals.intervalOperators import parse_bounds, wc_scalar_interval_feature
from ..pba.intervals.number import Interval
from numbers import Number
from ..pba.distributions import Distribution as pbaDistribution
import operator
from pint import Quantity


""" Uncertain Number class """


if TYPE_CHECKING:
    from ..pba.intervals.number import Interval
    from ..pba.distributions import Distribution as pbaDistribution


__all__ = [
    "UncertainNumber",
    "I",
    "D",
    "DSS",
    "norm",
    "gaussian",
    "gamma",
    "normal",
    "alpha",
    "anglit",
    "argus",
    "arcsine",
    "beta",
    "betaprime",
    "bradford",
    "burr",
    "burr12",
    "cauchy",
    "chi",
    "chi2",
    "cosine",
    "uniform",
    "lognormal",
]


class UncertainNumber:
    """Uncertain Number class

    args:
        - intervals (Interval): the interval specification for the UN object;

        - distribution_parameters: a list of the distribution family and its parameters; e.g. ['norm', [0, 1]];

        - pbox_initialisation: a list of the distribution family and its parameters; e.g. ['norm', ([0,1], [3,4])];


    example:

        Uncertain numbers can be constructed in multiple ways. For example, a canonical way allows users to fill in as many fields as possible:

        >>> from pyuncertainnumber import UncertainNumber
        >>> UncertainNumber(name="velocity", symbol="v", unit="m/s", intervals=[1, 2])
        >>> UncertainNumber(name="velocity", symbol="v", unit="m/s", distribution_parameters=['normal', (10, 2)])
        >>> UncertainNumber(name="velocity", symbol="v", unit="m/s", pbox_parameters=['normal', ([8, 12], [0.5, 1.5])])
        >>> UncertainNumber(name="velocity", symbol="v", unit="m/s", essence='dempster_shafer', intervals=[[1,5], [3,6]], masses=[0.5, 0.5])


        Alternatively, users can use shortcuts to quickly create UN objects and get on with calculations:

        >>> import pyuncertainnumber as pun
        >>> pun.I([1, 2])
        >>> pun.D('gaussian', (10, 2))
        >>> pun.normal([8, 12], [0.5, 1.5])
        >>> pun.DSS(intervals=[[1,5], [3,6]], masses=[0.5, 0.5])
    """

    Q_ = Quantity
    instances = []

    def __init__(
        self,
        name=None,
        symbol=None,
        unit=None,
        uncertainty_type=None,
        essence=None,
        masses=None,
        intervals=None,
        distribution_parameters=None,
        pbox_parameters=None,
        hedge=None,
        _construct=None,
        nominal_value=1.0,
        p_flag=True,
        _skip_construct_init=False,
        measurand=None,
        nature=None,
        provenence=None,
        justification=None,
        structure=None,
        security=None,
        ensemble=None,
        variability=None,
        dependence=None,
        uncertainty=None,
        physical_quantity=None,
        _samples=None,
        **kwargs,
    ):
        self.name = name
        self.symbol = symbol
        self.uncertainty_type = uncertainty_type
        self.essence = essence
        self.masses = masses
        self.intervals = intervals
        self.distribution_parameters = distribution_parameters
        self.pbox_parameters = pbox_parameters
        self.hedge = hedge
        self._construct = _construct
        self.nominal_value = nominal_value
        self.p_flag = p_flag
        self._skip_construct_init = _skip_construct_init
        self.measurand = measurand
        self.nature = nature
        self.provenence = provenence
        self.justification = justification
        self.structure = structure
        self.security = security
        self.ensemble = ensemble
        self.variability = variability
        self.dependence = dependence
        self.uncertainty = uncertainty
        self._physical_quantity = physical_quantity
        self._samples = _samples
        self.__init_check()
        self.__init_construct()
        self.nominal_value = self._construct.nominal_value
        self.unit = unit

    # *  ---------------------more on initialisation---------------------*#

    def __init_check(self):
        UncertainNumber.instances.append(self)

        if not self.essence:
            check_initialisation_list = [
                self.intervals,
                self.distribution_parameters,
                self.pbox_parameters,
            ]
            if any(v is not None for v in check_initialisation_list):
                raise ValueError(
                    "The 'essence' of the Uncertain Number is not specified"
                )
            if (self._construct is None) | (not self._construct):
                print("a vacuous interval is created")
                self.essence = "interval"
                self.intervals = [-np.inf, np.inf]

    def __init_construct(self):
        """the de facto parameterisation/instantiation procedure for the core constructs of the UN class

        caveat:
            user needs to by themselves figure out the correct
            shape of the 'distribution_parameters', such as ['uniform', [1,2]]
        """

        # * ------------------------ create the underlying construct

        if not self._skip_construct_init:
            match self.essence:
                case "interval":
                    self._construct = parse_bounds(self.intervals)
                case "distribution" | "pbox":
                    if self.pbox_parameters is not None:
                        par = Parameterisation(
                            self.pbox_parameters, essence=self.essence
                        )
                    else:
                        par = Parameterisation(
                            self.distribution_parameters, essence=self.essence
                        )
                    self._construct = par.yield_construct()
                case "dempster_shafer":
                    from ..pba.dss import DempsterShafer

                    self._construct = DempsterShafer(
                        intervals=parse_bounds(self.intervals), masses=self.masses
                    )

    def parameterised_pbox_specification(self):
        if self.p_flag:
            self._construct = self.match_pbox(
                self.distribution_parameters[0],
                self.distribution_parameters[1],
            )
            self.nominal_value = self._construct.mean().midpoint()

    def _update_physical_quantity(self):
        self._physical_quantity = self.Q_(self.nominal_value, self.unit)

    @staticmethod
    def match_pbox(keyword, parameters):
        """match the distribution keyword from the initialisation to create the underlying distribution object

        args:
            - keyword: (str) the distribution keyword
            - parameters: (list) the parameters of the distribution
        """
        obj = named_pbox.get(
            keyword, "You're lucky as the distribution is not supported"
        )
        if isinstance(obj, str):
            print(obj)  # print the error message
        return obj(*parameters)

    def init_check(self):
        """check if the UN initialisation specification is correct

        note:
            a lot of things to double check. keep an growing list:
            1. unit
            2. hedge: user cannot speficy both 'hedge' and 'intervals'. 'intervals' takes precedence.

        """
        pass

    # * ---------------------object representation---------------------* #

    def __str__(self):
        """string representation of the UncertainNumber

        note:
            the same as __reor__ for now until a better idea is proposed
        """
        return self.__repr__()

    def __repr__(self) -> str:
        """Concise __repr__"""

        field_values = get_concise_repr(self.__dict__)
        field_str = ", ".join(
            f"{k}={float(v)!r}" if isinstance(v, np.floating) else f"{k}={v!r}"
            for k, v in field_values.items()
        )

        # fancy string formatting of unit
        u_str = f", physical_quantity={self._physical_quantity:~P}"
        if not self._physical_quantity.dimensionless:
            field_str += u_str

        return f"{self.__class__.__name__}({field_str})"

    def describe(self, style="verbose"):
        """print out a verbose description of the uncertain number"""

        match style:
            case "verbose":
                match self.essence:
                    case "interval":
                        return f"This is an {self.essence}-type Uncertain Number whose min value is {self._construct.left:.2f} and max value is {self._construct.right:.2f}. An interval is a range of values that are possible for the measurand whose value is unknown, which typically represents the epistemic uncertainty. The interval is defined by the minimum and maximum values (i.e. lower bound and upper bound) that the measurand could take on."
                    case "distribution":
                        return f"This is a {self.essence}-type Uncertain Number that follows a {self.distribution_parameters[0]} distribution with parameters {self.distribution_parameters[1]}. Probability distributios are typically empolyed to model aleatoric uncertainty, which represents inherent randomness. The distribution is defined by the probability density function (pdf) or cumulative distribution function (cdf)."
                    case "pbox":
                        try:
                            return f"This is a {self.essence}-type Uncertain Number that follows a {self.pbox_parameters[0]} distribution with parameters {self.pbox_parameters[1]}"
                        except:
                            return f"This is a {self.essence}-type Uncertain Number that follows a {self.distribution_parameters[0]} distribution with parameters {self.distribution_parameters[1]}"

            case "one-number":
                return f"This is an {self.essence}-type Uncertain Number whose naked value is {self.nominal_value:.2f}"
            case "concise":
                return self.__repr__()
            case "range":
                match self.essence:
                    case "interval":
                        return f"This is an {self.essence}-type Uncertain Number whose min value is {self._construct.left:.2f} and max value is {self._construct.right:.2f}."
                    case "distribution":
                        return f"This is an {self.essence}-type Uncertain Number with 'some' range of {self._construct._range_list[0]:.2f} and {self._construct._range_list[1]:.2f}."
                    case "pbox":
                        return f"This is an {self.essence}-type Uncertain Number with 'some' range of {self._construct.left:.2f} and {self._construct.right:.2f}."
            case "five-number":
                match self.essence:
                    case "interval":
                        return f"This is an {self.essence}-type Uncertain Number that does not support this description."
                    case "distribution":
                        print(
                            f"This is an {self.essence}-type Uncertain Number whose statistical description is shown below:\n"
                            f"- family: {self.distribution_parameters[0]}\n"
                            f"- min: {self._construct._range_list[0]:.2f}\n"
                            f"- Q1: something\n"
                            f"- mean: {self._construct.mean_left}\n"
                            f"- Q3: something\n"
                            f"- variance: something"
                        )
            case "risk calc":
                match self.essence:
                    case "interval":
                        return "Will show a plot of the interval"
                    case "distribution":
                        print(
                            f"This is an {self.essence}-type Uncertain Number of family '{self.distribution_parameters[0]}' parameterised by {self.distribution_parameters[1]}"
                        )
                        self._construct.quick_plot()

    # * ---------------------some methods---------------------* #

    def ci(self):
        """get 95% range confidence interval"""
        match self.essence:
            case "interval":
                return [self._construct.left, self._construct.right]
            case "distribution":
                which_dist = self.distribution_parameters[0]
                if which_dist == "norm":
                    rv = norm(*self.distribution_parameters[1])
                    return [rv.ppf(0.025), rv.ppf(0.975)]
            case "pbox":
                return "unfinshed"

    def plot(self, **kwargs):
        """quick plot of the uncertain number object"""

        return self._construct.plot(**kwargs)

    def display(self, **kwargs):
        """quick plot of the uncertain number object"""

        return self._construct.display(**kwargs)

    # * ---------------------properties --------------------- *#
    @property
    def construct(self):
        return self._construct

    @property
    def construct_type(self):
        type(self._construct)

    @property
    def unit(self):
        """get the physical quantity of the uncertain number"""
        return self._units

    @unit.setter
    def unit(self, value):
        """set the physical quantity of the uncertain number"""
        self._units = value
        self._update_physical_quantity()

    @property
    def physical_quantity(self):
        """get the physical quantity of the uncertain number"""
        return self._physical_quantity

    @physical_quantity.setter
    def physical_quantity(self, value):
        """set the physical quantity of the uncertain number"""
        self._physical_quantity = value

    # * --------------------- constructors--------------------- *#

    @classmethod
    def from_hedge(cls, hedged_language):
        """create an Uncertain Number from hedged language"""
        an_obj = hedge_interpret(hedged_language)
        essence = "interval"  # TODO: choose between interval, pbox
        left, right = an_obj.left, an_obj.right
        return cls(essence=essence, intervals=[left, right])

    @classmethod
    def fromConstruct(cls, construct):
        """create an Uncertain Number from a construct object"""
        from ..pba.pbox_abc import Pbox
        from ..pba.dss import DempsterShafer
        from ..pba.distributions import Distribution as pbaDistribution

        if isinstance(construct, Pbox):
            return cls.from_pbox(construct)
        if isinstance(construct, Interval):
            return cls.from_Interval(construct)
        if isinstance(construct, DempsterShafer):
            return cls.from_ds(construct)
        if isinstance(construct, pbaDistribution):
            return cls.fromDistribution(construct)
        if isinstance(construct, cls):
            return construct
        if isinstance(
            construct, np.ndarray
        ):  # a fail-safe exit, which may be coerced into a Distribution UN later on
            return construct
        else:
            raise ValueError("The construct object is not recognised")

    @classmethod
    def fromDistribution(cls, D, **kwargs):
        # dist_family: str, dist_params,
        """create an Uncertain Number from a Distribution object.

        args:
            - D (Distribution): a Distribution object
            dist_family (str): the distribution family
            dist_params (list, tuple or string): the distribution parameters
        """
        # distSpec = DistributionSpecification(D.dist_family, D.dist_params)

        if D.empirical_data is None:
            return cls(
                essence="distribution",
                distribution_parameters=[D.dist_family, D.dist_params],
                **kwargs,
            )
        else:
            return cls(
                essence="distribution",
                distribution_parameters=None,
                _samples=D.sample_data,
            )

    @classmethod
    def from_Interval(cls, u):
        return cls(essence="interval", intervals=u)

    @classmethod
    def from_pbox(cls, p):
        """genenal from  pbox"""
        # passPboxParameters()
        return cls(
            essence="pbox", p_flag=False, _construct=p, _skip_construct_init=True
        )

    @classmethod
    def from_ds(cls, ds):
        return cls.from_pbox(ds.to_pbox())

    @classmethod
    def from_sps(cls, sps_dist):
        """create an UN object from a parametric scipy.stats dist object
        #! it seems that a function will suffice
        args:
            - sps_dist: scipy.stats dist object

        note:
            - sps_dist --> UN.Distribution object
        """
        pass

    # * ---------------------arithmetic operations---------------------#

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented
        if len(inputs) != 1 or inputs[0] is not self:
            return NotImplemented
        if "out" in kwargs and kwargs["out"] is not None:
            return NotImplemented

        if ufunc is np.sin:
            return self.sin()
        if ufunc is np.cos:
            return self.cos()
        if ufunc is np.tan:
            return self.tan()
        if ufunc is np.tanh:
            return self.tanh()
        if ufunc is np.exp:
            return self.exp()
        if ufunc is np.sqrt:
            return self.sqrt()
        if ufunc is np.log:
            return self.log()
        if ufunc is np.reciprocal:
            return self.reciprocal()

        return NotImplemented

    # * ---------------------unary operations---------------------#

    def _apply(self, method):
        from ..pba.operation import convert

        target = convert(self.construct)
        return UncertainNumber.fromConstruct(getattr(target, method)())

    def sqrt(self):
        return self._apply("sqrt")

    def exp(self):
        return self._apply("exp")

    def tanh(self):
        return self._apply("tanh")

    def tan(self):
        return self._apply("tan")

    def log(self):
        return self._apply("log")

    def sin(self):
        return self._apply("sin")

    def cos(self):
        return self._apply("cos")

    def reciprocal(self):
        return self._apply("reciprocal")

    # * ---------------------binary operations---------------------#

    def bin_ops(self, other, ops):
        from ..pba.pbox_abc import convert_pbox

        # new_cons = ops(self._construct, other._construct)
        if is_un(other) == 0:
            new_cons = ops(self._construct, other)
        elif is_un(other) == 1:
            a = convert_pbox(self._construct)
            b = convert_pbox(other._construct)
            new_cons = ops(a, b)
        elif is_un(other) == 2:
            raise ValueError(
                "construct object entered. propagation can be done but units won't be passed down"
            )

        new_un = UncertainNumber.fromConstruct(new_cons)
        return pass_down_units(self, other, ops, new_un)

    def __add__(self, other):
        """add two uncertain numbers"""
        return self.bin_ops(other, operator.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.bin_ops(other, operator.sub)

    def __mul__(self, other):
        """multiply two uncertain numbers"""
        return self.bin_ops(other, operator.mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """divide two uncertain numbers"""
        return self.bin_ops(other, operator.truediv)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __pow__(self, other):
        """power of two uncertain numbers"""
        return self.bin_ops(other, operator.pow)

    def __rpow__(self, other):
        """power of two uncertain numbers"""
        from ..pba.operation import convert

        return other ** convert(self.construct)

    # * ---------------------w/ dependency ---------------------#

    def add(self, other, dependency="f"):
        from ..pba.operation import convert

        a, b = convert(self.construct), convert(other.construct)
        return a.add(b, dependency=dependency)

    def sub(self, other, dependency="f"):
        from ..pba.operation import convert

        a, b = convert(self.construct), convert(other.construct)
        return a.sub(b, dependency=dependency)

    def mul(self, other, dependency="f"):
        from ..pba.operation import convert

        a, b = convert(self.construct), convert(other.construct)
        return a.mul(b, dependency=dependency)

    def div(self, other, dependency="f"):
        from ..pba.operation import convert

        a, b = convert(self.construct), convert(other.construct)
        return a.div(b, dependency=dependency)

    def pow(self, other, dependency="f"):
        from ..pba.operation import convert

        a, b = convert(self.construct), convert(other.construct)
        return a.pow(b, dependency=dependency)

    # * ---------------------serialisation functions---------------------*#

    def JSON_dump(self, filename="UN_data.json"):
        """the JSON serialisation of the UN object into the filesystem"""

        filepath = Path(Config.result_path) / filename
        with open(filepath, "w") as fp:
            json.dump(self, fp, cls=UNEncoder, indent=4)

    # * ---------------------other functions---------------------*#
    def to_pbox(self):
        """convert the UN object to a pbox object"""

        from ..pba.operation import convert

        pbox = convert(self._construct)
        return pbox


# * ---------------------shortcuts --------------------- *#
def makeUNPbox(func):
    """Constructor decorator to create a UN from a parametric pbox"""
    from ..pba.pbox_parametric import _bound_pcdf

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        family_str = func(*args, **kwargs)
        p = _bound_pcdf(family_str, *args)
        return UncertainNumber.fromConstruct(p)

    return wrapper_decorator


def constructUN(func):
    """from a construct to create a UN"""

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        p = func(*args, **kwargs)
        return UncertainNumber.fromConstruct(p)

    return wrapper_decorator


def I(*args: str | list[Number] | Interval) -> UncertainNumber:
    """a shortcut to construct the interval-type UN object"""
    return UncertainNumber.fromConstruct(wc_scalar_interval_feature(*args))


def D(*args, **kwargs) -> UncertainNumber:
    """a shortcut to construct the distribution-type UN object"""
    from ..pba.distributions import Distribution as pbaDistribution

    dist = pbaDistribution(*args, **kwargs)
    return UncertainNumber.fromDistribution(dist)


def DSS(*args, **kwargs) -> UncertainNumber:
    """a shortcut for the Dempster-Shafer-type UN object"""
    from ..pba.dss import DempsterShafer

    ds = DempsterShafer(*args, **kwargs)
    return UncertainNumber.from_ds(ds)


# * ---------------------parse inputs for UN only  --------------------- *#


def match_pbox(keyword, parameters):
    """match the distribution keyword from the initialisation to create the underlying distribution object

    args:
        - keyword: (str) the distribution keyword
        - parameters: (list) the parameters of the distribution
    """
    obj = named_pbox.get(keyword, "You're lucky as the distribution is not supported")
    if isinstance(obj, str):
        print(obj)  # print the error message
    return obj(*parameters)


class Parameterisation:
    """Parameterisation specification of the UN object

    args:
        - parm_specification (list): a combo of the distribution family and its parameters; e.g. ['norm', [0, 1]];
        - essence (str): 'distribution' or 'pbox'
    """

    def __init__(self, parm_specification: list, essence: str):
        self.parm_specification = ParamSpecification(
            parm_specification
        )  # combo e.g. ['norm', (0, 1)]
        self.essence = essence

    def yield_construct(self):
        if self.essence == "pbox":
            pbox = match_pbox(
                self.parm_specification.family, self.parm_specification.parameters
            )
            return pbox
        elif self.essence == "distribution":
            dist = pbaDistribution(
                dist_family=self.parm_specification.family,
                dist_params=self.parm_specification.parameters,
            )
            return dist


class ParamSpecification:
    """The combo specification of the distribution family and its parameters

    note:
        Only used for the format of specification.
        a recommended specification: ['gaussian', (12,4)] or ['gaussian', ([0,12],[1,4])]
    """

    def __init__(self, input):
        if (
            not isinstance(input, list)
            or len(input) != 2
            or not isinstance(input[0], str)
        ):
            raise ValueError("Input must be in the format ['str', (a, b)]")

        self.family = input[0]
        self.parameters = input[1]
        # self.un_type_check()

    def supported_distribution_check(self):
        """check if the family is implemented"""
        pass

    def un_type_check(self):
        """infer the real type of UN given the specification
        # NOT in USE. lousy logic
        """

        # distribution case
        if all(isinstance(x, Number) for x in self.parameters):
            self._true_type = "distribution"
        else:
            self._true_type = "pbox"


def pass_down_units(a, b, ops, t):
    """pass down the unit of the uncertain number

    args:
        - a: the first uncertain number
        - b: the second uncertain number
        - ops: the operation to be performed
        - t: the result uncertain number of the operation
    """
    if is_un(b) == 0:
        try:
            new_q = ops(a._physical_quantity, b * a._physical_quantity.units)
        except Exception:
            new_q = ops(a._physical_quantity, b)
    elif is_un(b) == 1:
        new_q = ops(a._physical_quantity, b._physical_quantity)

    t.physical_quantity = new_q
    return t


def is_un(sth):
    """utility function to decide the essence of the object

    returns:
        - 0: if sth is a regular number, float or int
        - 1: if sth is an UncertainNumber object
        - 2: if sth is a construct in {Interval, Pbox, DempsterShafer, or Distribution}
    """

    from ..pba.distributions import Distribution
    from ..pba.dss import DempsterShafer
    from ..pba.pbox_abc import Pbox
    from ..pba.intervals.number import Interval

    if isinstance(sth, Number):
        return 0
    elif isinstance(sth, UncertainNumber):
        return 1
    elif isinstance(sth, Interval | Pbox | DempsterShafer | Distribution):
        return 2


def exist_un(a_list) -> bool:
    """check if there is any UN object in the list"""
    return any(is_un(x) == 1 for x in a_list)


# * ---------------------parametric shortcuts  --------------------- *#


@makeUNPbox
def norm(*args):
    return "norm"


@makeUNPbox
def gaussian(*args):
    return "norm"


@makeUNPbox
def normal(*args):
    return "norm"


@makeUNPbox
def alpha(*args):
    return "alpha"


@makeUNPbox
def anglit(*args):
    return "anglit"


@makeUNPbox
def argus(*args):
    return "argus"


@makeUNPbox
def arcsine(*args):
    return "arcsine"


@makeUNPbox
def beta(*args):
    return "beta"


@makeUNPbox
def betaprime(*args):
    return "betaprime"


@makeUNPbox
def bradford(*args):
    return "bradford"


@makeUNPbox
def burr(*args):
    return "burr"


@makeUNPbox
def burr12(*args):
    return "burr12"


@makeUNPbox
def cauchy(*args):
    return "cauchy"


@makeUNPbox
def chi(*args):
    return "chi"


@makeUNPbox
def chi2(*args):
    return "chi2"


@makeUNPbox
def cosine(*args):
    return "cosine"


@makeUNPbox
def crystalball(*args):
    return "crystalball"


@makeUNPbox
def dgamma(*args):
    return "dgamma"


@makeUNPbox
def dweibull(*args):
    return "dweibull"


@makeUNPbox
def erlang(*args):
    return "erlang"


@makeUNPbox
def expon(*args):
    return "expon"


@makeUNPbox
def exponnorm(*args):
    return "exponnorm"


@makeUNPbox
def exponweib(*args):
    return "exponweib"


@makeUNPbox
def exponpow(*args):
    return "exponpow"


@makeUNPbox
def f(*args):
    return "f"


@makeUNPbox
def fatiguelife(*args):
    return "fatiguelife"


@makeUNPbox
def fisk(*args):
    return "fisk"


@makeUNPbox
def foldcauchy(*args):
    return "foldcauchy"


@makeUNPbox
def genlogistic(*args):
    return "genlogistic"


@makeUNPbox
def gennorm(*args):
    return "gennorm"


@makeUNPbox
def genpareto(*args):
    return "genpareto"


@makeUNPbox
def genexpon(*args):
    return "genexpon"


@makeUNPbox
def genextreme(*args):
    return "genextreme"


@makeUNPbox
def gausshyper(*args):
    return "gausshyper"


@makeUNPbox
def gamma(*args):
    return "gamma"


@makeUNPbox
def gengamma(*args):
    return "gengamma"


@makeUNPbox
def genhalflogistic(*args):
    return "genhalflogistic"


@makeUNPbox
def geninvgauss(*args):
    return "geninvgauss"


@makeUNPbox
def gompertz(*args):
    return "gompertz"


@makeUNPbox
def gumbel_r(*args):
    return "gumbel_r"


@makeUNPbox
def gumbel_l(*args):
    return "gumbel_l"


@makeUNPbox
def halfcauchy(*args):
    return "halfcauchy"


@makeUNPbox
def halflogistic(*args):
    return "halflogistic"


@makeUNPbox
def halfnorm(*args):
    return "halfnorm"


@makeUNPbox
def halfgennorm(*args):
    return "halfgennorm"


@makeUNPbox
def hypsecant(*args):
    return "hypsecant"


@makeUNPbox
def invgamma(*args):
    return "invgamma"


@makeUNPbox
def invgauss(*args):
    return "invgauss"


@makeUNPbox
def invweibull(*args):
    return "invweibull"


@makeUNPbox
def irwinhall(*args):
    return "irwinhall"


@makeUNPbox
def jf_skew_t(*args):
    return "jf_skew_t"


@makeUNPbox
def johnsonsb(*args):
    return "johnsonsb"


@makeUNPbox
def johnsonsu(*args):
    return "johnsonsu"


@makeUNPbox
def kappa4(*args):
    return "kappa4"


@makeUNPbox
def kappa3(*args):
    return "kappa3"


@makeUNPbox
def ksone(*args):
    return "ksone"


@makeUNPbox
def kstwo(*args):
    return "kstwo"


@makeUNPbox
def kstwobign(*args):
    return "kstwobign"


@makeUNPbox
def laplace(*args):

    return "laplace"


@makeUNPbox
def laplace_asymmetric(*args):
    return "laplace_asymmetric"


@makeUNPbox
def levy(*args):
    return "levy"


@makeUNPbox
def levy_l(*args):
    return "levy_l"


@makeUNPbox
def levy_stable(*args):
    return "levy_stable"


@makeUNPbox
def logistic(*args):
    return "logistic"


@makeUNPbox
def loggamma(*args):
    return "loggamma"


@makeUNPbox
def loglaplace(*args):
    return "loglaplace"


@makeUNPbox
def loguniform(*args):
    return "loguniform"


@makeUNPbox
def lomax(*args):
    return "lomax"


@makeUNPbox
def maxwell(*args):
    return "maxwell"


@makeUNPbox
def mielke(*args):
    return "mielke"


@makeUNPbox
def moyal(*args):
    return "moyal"


@makeUNPbox
def nakagami(*args):
    return "nakagami"


@makeUNPbox
def ncx2(*args):
    return "ncx2"


@makeUNPbox
def ncf(*args):
    return "ncf"


@makeUNPbox
def nct(*args):
    return "nct"


@makeUNPbox
def norminvgauss(*args):
    return "norminvgauss"


@makeUNPbox
def pareto(*args):
    return "pareto"


@makeUNPbox
def pearson3(*args):
    return "pearson3"


@makeUNPbox
def powerlaw(*args):
    return "powerlaw"


@makeUNPbox
def powerlognorm(*args):
    return "powerlognorm"


@makeUNPbox
def powernorm(*args):
    return "powernorm"


@makeUNPbox
def rdist(*args):
    return "rdist"


@makeUNPbox
def rayleigh(*args):
    return "rayleigh"


@makeUNPbox
def rel_breitwigner(*args):
    return "rel_breitwigner"


@makeUNPbox
def rice(*args):
    return "rice"


@makeUNPbox
def recipinvgauss(*args):
    return "recipinvgauss"


@makeUNPbox
def semicircular(*args):
    return "semicircular"


@makeUNPbox
def skewcauchy(*args):
    return "skewcauchy"


@makeUNPbox
def skewnorm(*args):
    return "skewnorm"


@makeUNPbox
def studentized_range(*args):
    return "studentized_range"


@makeUNPbox
def t(*args):
    return "t"


@makeUNPbox
def trapezoid(*args):
    return "trapezoid"


@makeUNPbox
def triang(*args):
    return "triang"


@makeUNPbox
def truncexpon(*args):
    return "truncexpon"


@makeUNPbox
def truncpareto(*args):
    return "truncpareto"


@makeUNPbox
def truncweibull_min(*args):
    return "truncweibull_min"


@makeUNPbox
def tukeylambda(*args):
    return "tukeylambda"


def uniform_sps(*args):
    return "uniform"


@makeUNPbox
def vonmises(*args):
    return "vonmises"


@makeUNPbox
def vonmises_line(*args):
    return "vonmises_line"


@makeUNPbox
def wald(*args):
    return "wald"


@makeUNPbox
def weibull_min(*args):
    return "weibull_min"


@makeUNPbox
def weibull_max(*args):
    return "weibull_max"


@makeUNPbox
def wrapcauchy(*args):
    return "wrapcauchy"


# *---------------------some special ones ---------------------*#

from ..pba.pbox_parametric import lognormal, uniform

uniform = constructUN(uniform)
lognormal = constructUN(lognormal)
