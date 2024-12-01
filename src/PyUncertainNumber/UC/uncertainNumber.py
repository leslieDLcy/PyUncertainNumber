from dataclasses import dataclass, field
from typing import Type, Union

# from .measurand import Measurand
# from .variability import Variability
from .uncertainty_types import Uncertainty_types
from .ensemble import Ensemble
from PyUncertainNumber.pba.interval import Interval as nInterval
from PyUncertainNumber.pba.interval import PM
from .utils import *
from .params import Params
from typing import List
from pint import UnitRegistry
from pathlib import Path
from PyUncertainNumber.UP.vertex import vertexMethod as vM
from PyUncertainNumber.UP.endpoints import endpoints_propagation_2n
from PyUncertainNumber.nlp.language_parsing import hedge_interpret
from scipy.stats import norm
from .check import DistributionSpecification
from PyUncertainNumber.pba.pbox import named_pbox
from typing import Sequence
from functools import singledispatch
from .intervalOperators import wc_interval


""" Uncertain Number class """


@dataclass
class UncertainNumber:
    """Uncertain Number class

    args:
        - `bounds`;
        - `distribution_parameters`: a list of the distribution family and its parameters; e.g. ['norm', [0, 1]];
        - `pbox_initialisation`: a list of the distribution family and its parameters; e.g. ['norm', ([0,1], [3,4])];
        -  naked_value: the deterministic numeric representation of the UN object, which shall be linked with the 'pba' or `Intervals` package
    """

    # ---------------------Basic---------------------#
    name: str = field(default=None)
    symbol: str = field(default=None)
    units: str = field(default=None)

    # ---------------------Value---------------------#
    # ensemble: Type[Ensemble] = field(default=None)
    uncertainty_type: Type[Uncertainty_types] = field(default=None)
    essence: str = field(default=None)  # [interval, distribution, pbox]
    bounds: Union[List[float], str] = field(default=None)
    distribution_parameters: list[str, float | int] = field(default=None)
    pbox_parameters: list[str, Sequence[nInterval]] = field(
        default=None)
    hedge: str = field(default=None)
    # this is the deterministic numeric representation of the
    # UN object, which shall be linked with the 'pba' or `Intervals` package
    naked_value: float = field(default=None)

    # ---------------------auxlliary information---------------------#
    # some simple boiler plates
    # lat: float = field(default=0.0, metadata={'unit': 'degrees'})
    # ensemble: Type[Ensemble] = field(default=None)

    measurand: str = field(default=None)
    nature: str = field(default=None)
    provenence: str = field(default=None)
    justification: str = field(default=None)
    structure: str = field(default=None)
    security: str = field(default=None)

    # ---------------------aleatoric component---------------------#
    ensemble: Type[Ensemble] = field(default=None)
    variability: str = field(default=None)
    dependence: str = field(default=None)

    # ---------------------epistemic component---------------------#
    uncertainty: str = field(default=None)

    # class variable
    instances = []  # TODO named as registry later on

    # ---------------------more on initialisation---------------------#

    def __post_init__(self):
        """the de facto initialisation method for the core math objects of the UN class

        caveat:
            user needs to by themselves figure out the correct
            shape of the 'distribution_parameters', such as ['uniform', [1,2]]
        """

        if not self.essence:
            check_initialisation_list = [
                self.bounds,
                self.distribution_parameters,
                self.pbox_parameters,
            ]
            if any(v is not None for v in check_initialisation_list):
                raise ValueError(
                    "The 'essence' of the Uncertain Number is not specified"
                )
            else:
                print("a vacuous interval is created")
                self.essence = "interval"
                self.bounds = [-np.inf, np.inf]

        # TODO to create the Quantity object defined as <value * unit>
        """get the 'unit' representation of the uncertain number"""
        ureg = UnitRegistry()
        UncertainNumber.instances.append(self)
        # I can use the following logic to double check the arithmetic operations of the UN object
        # if isinstance(self.naked_value, float) or isinstance(
        #     self.naked_value, int
        # ):
        #     self._UnitsRep = self.naked_value * ureg(self.units)
        # else:
        #     self._UnitsRep = 1 * ureg(self.units)

        self._UnitsRep = 1 * ureg(self.units)

        ### create the underlying construct ###
        match self.essence:
            case "interval":
                self._construct = _parse_bounds(self.bounds)
                self.naked_value = self._construct.midpoint()
            case "distribution":
                self._construct = self.match_distribution(
                    self.distribution_parameters[0],
                    self.distribution_parameters[1],
                )
                self.naked_value = (
                    self._construct.mean_left
                )  # TODO the error is here where the numeric value is NOT a value
                # TODO continue getting familar with the 'pba' package for computing mean etc...
                # TODO I've put `mean_left` there but unsure if correct or not
            case "pbox":
                self._construct = self.match_distribution(
                    self.distribution_parameters[0],
                    self.distribution_parameters[1],
                )
                self.naked_value = self._construct.mean()

    @staticmethod
    def match_distribution(keyword, parameters):
        """match the distribution keyword from the initialisation to create the underlying distribution object

        args:
            - keyword: (str) the distribution keyword
            - parameters: (list) the parameters of the distribution
        """

        obj = named_pbox.get(keyword,
                             "You're lucky as the distribution is not supported")
        if isinstance(obj, str):
            print(obj)  # print the error message
        return obj(*parameters)

    def init_check(self):
        """check if the UN initialisation specification is correct

        note:
            a lot of things to double check. keep an growing list:
            1. unit
            2. hedge: user cannot speficy both 'hedge' and 'bounds'. 'bounds' takes precedence.

        """
        pass

    ##### object representations #####

    def __str__(self):
        """the verbose user-friendly string representation
        note:
            this has nothing to do with the logic of JSON serialisation
            ergo, do whatever you fancy;
        """
        field_values = {k: v for k, v in self.__dict__.items()
                        if v is not None}
        field_str = ", ".join(f"{k}={repr(v)}" for k,
                              v in field_values.items())
        return f"{self.__class__.__name__}({field_str})"

    def __repr__(self) -> str:
        """concise __repr__"""
        self._field_str = self._get_concise_representation()
        return f"{self.__class__.__name__}({self._field_str})"

    def describe(self, type="verbose"):
        """print out a verbose description of the uncertain number"""

        match type:
            case "verbose":
                match self.essence:
                    case "interval":
                        return f"This is an {self.essence}-type Uncertain Number whose min value is {self._construct.left:.2f} and max value is {self._construct.right:.2f}. An interval is a range of values that are possible for the measurand whose value is unknown, which typically represents the epistemic uncertainty. The interval is defined by the minimum and maximum values (i.e. lower bound and upper bound) that the measurand could take on."
                    case "distribution":
                        return f"This is a {self.essence}-type Uncertain Number that follows a {self.distribution_parameters[0]} distribution with parameters {self.distribution_parameters[1]}. Probability distributios are typically empolyed to model aleatoric uncertainty, which represents inherent randomness. The distribution is defined by the probability density function (pdf) or cumulative distribution function (cdf)."
                    case "pbox":
                        return f"This is a {self.essence}-type Uncertain Number that follows a {self.distribution_parameters[0]} distribution with parameters {self.distribution_parameters[1]}"
            case "one-number":
                return f"This is an {self.essence}-type Uncertain Number whose naked value is {self.naked_value:.2f}"
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

    # ---------------------some class methods---------------------#

    def _get_concise_representation(self):
        """get a concise representation of the UN object"""

        field_values = get_concise_repr(self.__dict__)
        return ", ".join(f"{k}={repr(v)}" for k, v in field_values.items())

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

    def display(self, **kwargs):
        """quick plot of the uncertain number object"""

        return self._construct.display(**kwargs)

    # ---------------------other constructors---------------------#

    @classmethod
    def from_hedge(cls, hedged_language):
        """create an Uncertain Number from hedged language

        note:
            # if interval or pbox, to be implemented later on
            #  currently only Interval is supported
        """
        an_obj = hedge_interpret(hedged_language)
        essence = "interval"  # TODO: choose between interval, pbox
        left, right = an_obj.left, an_obj.right
        return cls(essence=essence, bounds=[left, right])

    @classmethod
    def from_distribution(cls, dist_family: str, dist_params, **kwargs):
        """create an Uncertain Number from specification of distribution

        args:
            dist_family: str
                the distribution family
            dist_params: list, tuple or string
                the distribution parameters
        """
        distSpec = DistributionSpecification(dist_family, dist_params)
        if "essence" not in kwargs:
            kwargs["essence"] = "distribution"
        return cls(
            distribution_parameters=distSpec.get_specification(),
            **kwargs,
        )

    @classmethod
    def from_distributionProperties(cls, min, max, mean, median, variance, **kwargs):
        """to construct a pbox given the properties of the distribution

        returns:
            - a pbox-type UN object
        note:
            - whether differentiate explicitly if free/parametric pbox
        """
        pass

    @classmethod
    def I(cls, bounds, **kwargs):
        """a shortcut for creating an interval-type Uncertain Number"""
        return cls(essence="interval", bounds=bounds, **kwargs)

    # ---------------------arithmetic operations---------------------#

    def __add__(self, other):
        """add two uncertain numbers"""
        left = (self._construct + other._construct).left
        right = (self._construct + other._construct).right
        essence = self.essence
        return type(self)(essence=essence, bounds=[left, right])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """subtract two uncertain numbers"""
        left = (self._construct - other._construct).left
        right = (self._construct - other._construct).right
        essence = self.essence
        return type(self)(essence=essence, bounds=[left, right])

    def __mul__(self, other):
        """multiply two uncertain numbers"""

        if isinstance(other, int | float):
            left = (self._construct * other).left
            right = (self._construct * other).right
            essence = self.essence
            return type(self)(essence=essence, bounds=[left, right])
        elif isinstance(other, UncertainNumber):
            left = (self._construct * other._construct).left
            right = (self._construct * other._construct).right
            essence = self.essence
            return type(self)(essence=essence, bounds=[left, right])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """divide two uncertain numbers"""

        if isinstance(other, int | float):
            left = (self._construct / other).left
            right = (self._construct / other).right
            essence = self.essence
            return type(self)(essence=essence, bounds=[left, right])
        elif isinstance(other, UncertainNumber):
            left = (self._construct / other._construct).left
            right = (self._construct / other._construct).right
            essence = self.essence
            return type(self)(essence=essence, bounds=[left, right])

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __pow__(self, other):
        """power of two uncertain numbers"""

        if isinstance(other, int | float):
            left = (self._construct**other).left
            right = (self._construct**other).right
            essence = self.essence
            return type(self)(essence=essence, bounds=[left, right])
        elif isinstance(other, UncertainNumber):
            left = (self._construct**other._construct).left
            right = (self._construct**other._construct).right
            essence = self.essence
            return type(self)(essence=essence, bounds=[left, right])

    @classmethod
    def _toIntervalBackend(cls, vars=None) -> np.array:
        """transform any UN object to an `interval`
        #! currently in use
        # TODO think if use Marco's Interval Vector object

        question:
            - what is the `interval` representation: list, nd.array or Interval object?

        returns:
            - 2D np.array representation for all the interval-typed UNs
        """
        all_objs = {instance.symbol: instance for instance in cls.instances}

        if vars is not None:
            selected_objs = [all_objs[k] for k in all_objs if k in vars]
        else:
            selected_objs = [all_objs[k] for k in all_objs]

        # keep the order of the vars ....
        def as_interval(sth):
            """a helper function to convert to intervals"""
            if sth.essence == "interval":
                return sth.bounds
            else:
                return sth._construct.rangel

        _UNintervals_list = [as_interval(k) for k in selected_objs]
        _UNintervals = np.array(_UNintervals_list).reshape(-1, 2)
        return _UNintervals

    @classmethod
    def _IntervaltoCompBackend(cls, vars):
        """convert the interval-tupe UNs instantiated to the computational backend

        note:
            - it will automatically convert all the UN objects in array-like to the computational backend
            - essentially vars shall be all interval-typed UNs by now;

        returns:
            - nd.array or Marco's Interval object

        thoughts:
            - if Marco's, then we'd use `intervalise` func to get all interval objects
            and then to create another func to convert the interval objects to np.array to do endpoints method
        """

        # from augument list to intervals list
        all_objs = {instance.symbol: instance for instance in cls.instances}
        _intervals = [all_objs[k].bounds for k in all_objs if k in vars]
        _UNintervals = np.array(_intervals).reshape(-1, 2)
        return _UNintervals

    # ---------------------Uncertainty propatation methods---------------------#

    @classmethod
    def vertexMethod(cls, vars, func):
        """implementation of the endpoints method for the uncertain number

        args:
            vars: list
                the selected list of the symbols of UN or a list of arrays
            func: function
                the function to be applied to the uncertain number
        """

        if isinstance(vars[0], str):
            # _UNintervals = UncertainNumber._IntervaltoCompBackend(vars) # bp
            _UNintervals = UncertainNumber._toIntervalBackend(vars)
            df = vM(_UNintervals, func)
            return df
        elif isinstance(vars[0], int | float):
            # create a list of UN objects using hedge interpretation
            def get_hedgedUN(a_num_list):
                return [cls.from_hedge(f"{i}") for i in a_num_list]

            UN_list = get_hedgedUN(vars)
            _UNintervals = [k.bounds for k in UN_list]
            _UNintervals = np.array(_UNintervals).reshape(-1, 2)

            df = vM(_UNintervals, func)

            return df

    @classmethod
    def endpointsMethod(cls, vars, func, **kwargs):
        """implementation of the endpoints method for the uncertain number using
        Marco's implementation

        note:
            `vars` shall be consistent with the signature of `func`. This means that
            only a selected list of uncertain numbers will be used according to the func provided.

        args:
            vars: list
                the chosen list of uncertain numbers
            func: function
                the function to be applied to the uncertain number
        """
        # _UNintervals = UncertainNumber._IntervaltoCompBackend(vars) # bp
        _UNintervals = UncertainNumber._toIntervalBackend(vars)
        output_bounds_lo, output_bounds_hi, _, _ = endpoints_propagation_2n(
            _UNintervals, func
        )
        return cls(
            essence="interval",
            bounds=(output_bounds_lo, output_bounds_hi),
            **kwargs,
        )
        # return endpoints_propagation_2n(_UNintervals, func)

    # ---------------------serialisation functions---------------------#

    def JSON_dump(self, filename="UN_data.json"):
        """the JSON serialisation of the UN object into the filesystem"""

        filepath = Path(Params.result_path) / filename
        with open(filepath, "w") as fp:
            json.dump(self, fp, cls=UNEncoder, indent=4)


# ---------------------class related methods---------------------#

# TODO unfinished logic: currently if suffices in creating only `Interval` object
# @classmethod

    # def __add__(self, other):
    #     """ Add two uncertain numbers.
    #     #TODO unfinished logic for adding uncertain numbers
    # ! this code is kept for working with units
    #     """

    #     if not isinstance(other, type(self)):
    #         raise TypeError(
    #             "unsupported operand for +: "
    #             f"'{type(self).__name__}' and '{type(other).__name__}'"
    #         )
    #     if not self.unit == other.unit:
    #         raise TypeError(
    #             f"incompatible units: '{self.unit}' and '{other.unit}'"
    #         )

    #     return type(self)(super().__add__(other), self.unit)

def parse_description(description):
    # TODO add functionality for pbox
    """Parse the description of the uncertain number when initialising an Uncertain Number object

    args:
        description: str
            the flexible string desired by Scott to instantiate a Uncertain Number

    caveat:
        the description needs to have space between the values and the operators, such as '[15 +- 10%]'
    """

    ### type 1 ###
    # initial check if string-rep of list
    if initial_list_checking(description):
        an_int = initial_list_checking(description)
        if len(an_int) == 1:
            return PM(an_int[0], hw=Params.hw)
        elif len(an_int) > 1:
            return nInterval(*an_int)
    ### type 2 ###
    elif bad_list_checking(description):
        if PlusMinus_parser(description) & (not percentage_finder(description)):
            parsed_list = parser4(description)
            return PM(*parsed_list)
        elif PlusMinus_parser(description) & percentage_finder(description):
            # parse the percentage first
            mid_range = percentage_converter(description)
            parsed_mid_value = parser4(description)[0]

            # if we take the percentage literally
            # return PM(parsed_mid_value, hw=mid_range)
            # if we take the percentage based on the context
            return PM(parsed_mid_value, hw=parsed_mid_value * mid_range)


def _parse_interverl_inputs(vars):
    """ Parse the input intervals

    note:
        - Ioanna's funcs typically take 2D NumPy arra
    """
    if isinstance(vars, np.ndarray):
        if vars.shape[1] != 2:
            raise ValueError(
                "vars must be a 2D array with two columns per row (lower and upper bounds)")
        else:
            return vars

    if isinstance(vars, list):
        return UncertainNumber._toIntervalBackend(vars)


@singledispatch
def _parse_bounds(bounds):
    """ parse the self.bounds argument """
    return wc_interval(bounds)


@_parse_bounds.register(str)
def _str(bounds: str):
    return parse_description(bounds)
