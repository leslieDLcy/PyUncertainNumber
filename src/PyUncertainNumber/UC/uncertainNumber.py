from dataclasses import dataclass, field
from typing import Type, Union
from .measurand import Measurand
from .variability import Variability
from .uncertainty_types import Uncertainty_types
from .ensemble import Ensemble
from PyUncertainNumber.pba.interval import Interval as I
from PyUncertainNumber.pba.interval import PM
from PyUncertainNumber import pba
from .utils import *
from .params import Params
from typing import List
from pint import UnitRegistry
from pathlib import Path
from PyUncertainNumber.UP.vertex import vertexMethod as vM
from PyUncertainNumber.UP.endpoints import endpoints_propagation_2n
from PyUncertainNumber.NLP_constructor.language_parsing import hedge_interpret
from scipy.stats import norm
from .check import DistributionSpecification


""" Uncertain Number class """


@dataclass
class UncertainNumber:
    name: str = field(default=None)
    symbol: str = field(default=None)
    units: str = field(default=None)
    measurand: Type[Measurand] = field(default=None)
    variability: Type[Variability] = field(default=None)
    ensemble: Type[Ensemble] = field(default=None)
    uncertainty_type: Type[Uncertainty_types] = field(default=None)
    essence: str = field(default=None)  # [interval, distribution, pbox]
    interval_initialisation: Union[List[float], str] = field(default=None)
    distribution_initialisation: List[float] = field(default=None)
    pbox_initialisation: List[float] = field(default=None)  # must be 2D array
    hedge: str = field(default=None)
    # this is the deterministic numeric representation of the
    # UN object, which shall be linked with the 'pba' or `Intervals` package
    deter_value_rep: float = field(default=None)

    # a simple boiler plate
    # lat: float = field(default=0.0, metadata={'unit': 'degrees'})
    justification: str = field(default=None)

    # TODO add more metadata for the fields

    instances = []

    def __post_init__(self):
        """the de facto initialisation method for the core math objects of the UN class

        caveat:
            user needs to by themselves figure out the correct
            shape of the 'distribution_initialisation', such as ['uniform', [1,2]]
        """

        """get the 'unit' representation of the uncertain number"""
        ureg = UnitRegistry()
        UncertainNumber.instances.append(self)
        # I can use the following logic to double check the arithmetic operations of the UN object
        # if isinstance(self.deter_value_rep, float) or isinstance(
        #     self.deter_value_rep, int
        # ):
        #     self._UnitsRep = self.deter_value_rep * ureg(self.units)
        # else:
        #     self._UnitsRep = 1 * ureg(self.units)

        self._UnitsRep = 1 * ureg(self.units)

        # temp logic for parsing `self.interval_initialisation` if it is a string

        # create the underlying math object
        match self.essence:
            case "interval":
                # the default way of instantiating
                if isinstance(self.interval_initialisation, str):
                    self._math_object = parse_description(self.interval_initialisation)
                elif isinstance(self.interval_initialisation, list):
                    self._math_object = I(*self.interval_initialisation)
                elif isinstance(self.interval_initialisation, tuple):
                    self._math_object = I(*self.interval_initialisation)
            case "distribution":
                self._math_object = self.match_distribution(
                    self.distribution_initialisation[0],
                    self.distribution_initialisation[1],
                )
            case "pbox":
                self._math_object = self.match_distribution(
                    self.distribution_initialisation[0],
                    self.distribution_initialisation[1],
                )

        """create a deterministic representation of the uncertain number"""
        match self.essence:
            case "interval":
                self.deter_value_rep = self._math_object.midpoint()
            case "distribution":
                self.deter_value_rep = (
                    self._math_object.mean_left
                )  # TODO the error is here where the numeric value is NOT a value
                # TODO continue getting familar with the 'pba' package for computing mean etc...
                # TODO I've put `mean_left` there but unsure if correct or not
            case "pbox":
                self.deter_value_rep = self._math_object.mean()

    ##### object representations #####

    def __str__(self):
        """the verbose user-friendly string representation
        note:
            this has nothing to do with the logic of JSON serialisation
            ergo, do whatever you fancy;
        """
        field_values = {k: v for k, v in self.__dict__.items() if v is not None}
        field_str = ", ".join(f"{k}={repr(v)}" for k, v in field_values.items())
        return f"{self.__class__.__name__}({field_str})"

    """ new """

    def __repr__(self) -> str:
        """concise __repr__"""
        self._field_str = self._get_concise_representation()
        return f"{self.__class__.__name__}({self._field_str})"

    def description(self, type="verbose"):
        """print out a verbose description of the uncertain number"""

        match type:
            case "verbose":
                match self.essence:
                    case "interval":
                        return f"This is an {self.essence}-type Uncertain Number whose min value is {self._math_object.left:.2f} and max value is {self._math_object.right:.2f}. An interval is a range of values that are possible for the measurand whose value is unknown, which typically represents the epistemic uncertainty. The interval is defined by the minimum and maximum values (i.e. lower bound and upper bound) that the measurand could take on."
                    case "distribution":
                        return f"This is a {self.essence}-type Uncertain Number that follows a {self.distribution_initialisation[0]} distribution with parameters {self.distribution_initialisation[1]}. Probability distributios are typically empolyed to model aleatoric uncertainty, which represents inherent randomness. The distribution is defined by the probability density function (pdf) or cumulative distribution function (cdf)."
                    case "pbox":
                        return f"This is a {self.essence}-type Uncertain Number that follows a {self.distribution_initialisation[0]} distribution with parameters {self.distribution_initialisation[1]}"
            case "one-number":
                return f"This is an {self.essence}-type Uncertain Number whose naked value is {self.deter_value_rep:.2f}"
            case "concise":
                return self.__repr__()
            case "range":
                match self.essence:
                    case "interval":
                        return f"This is an {self.essence}-type Uncertain Number whose min value is {self._math_object.left:.2f} and max value is {self._math_object.right:.2f}."
                    case "distribution":
                        return f"This is an {self.essence}-type Uncertain Number with 'some' range of {self._math_object._range_list[0]:.2f} and {self._math_object._range_list[1]:.2f}."
                    case "pbox":
                        return f"This is an {self.essence}-type Uncertain Number with 'some' range of {self._math_object.left:.2f} and {self._math_object.right:.2f}."
            case "five-number":
                match self.essence:
                    case "interval":
                        return f"This is an {self.essence}-type Uncertain Number that does not support this description."
                    case "distribution":
                        print(
                            f"This is an {self.essence}-type Uncertain Number whose statistical description is shown below:\n"
                            f"- family: {self.distribution_initialisation[0]}\n"
                            f"- min: {self._math_object._range_list[0]:.2f}\n"
                            f"- Q1: something\n"
                            f"- mean: {self._math_object.mean_left}\n"
                            f"- Q3: something\n"
                            f"- variance: something"
                        )
            case "risk calc":
                match self.essence:
                    case "interval":
                        return "Will show a plot of the interval"
                    case "distribution":
                        print(
                            f"This is an {self.essence}-type Uncertain Number of family '{self.distribution_initialisation[0]}' parameterised by {self.distribution_initialisation[1]}"
                        )
                        self._math_object.quick_plot()

    @staticmethod
    def match_distribution(keyword, parameters):
        match keyword:
            case "normal":
                return pba.N(*parameters, steps=Params.steps)
            case "uniform":
                return pba.U(*parameters, steps=Params.steps)
            case _:
                print("Bad distribution specification")

    ##############################
    #######    some methods  #####
    ##############################

    def init_check(self):
        """check if the UN initialisation specification is correct

        note:
            a lot of things to double check. keep an growing list:
            1. unit
            2. hedge: user cannot speficy both 'hedge' and 'interval_initialisation'. 'interval_initialisation' takes precedence.

        """
        pass

    def _get_concise_representation(self):
        """get a concise representation of the UN object"""

        field_values = get_concise_repr(self.__dict__)
        return ", ".join(f"{k}={repr(v)}" for k, v in field_values.items())

    def ci(self):
        """get 95% range confidence interval"""
        match self.essence:
            case "interval":
                return [self._math_object.left, self._math_object.right]
            case "distribution":
                which_dist = self.distribution_initialisation[0]
                if which_dist == "norm":
                    rv = norm(*self.distribution_initialisation[1])
                    return [rv.ppf(0.025), rv.ppf(0.975)]
            case "pbox":
                return "unfinshed"

    ##############################
    ##### other constructors #####
    ##############################

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
        return cls(essence=essence, interval_initialisation=[left, right])

    @classmethod
    def from_distribution_setup(cls, dist_family, dist_params, **kwargs):
        distSpec = DistributionSpecification(dist_family, dist_params)
        if "essence" not in kwargs:
            kwargs["essence"] = "distribution"
        return cls(
            distribution_initialisation=distSpec.get_specification(),
            **kwargs,
        )

    ##############################
    ##### arithmetic operations ##
    ##############################

    def __add__(self, other):
        """add two uncertain numbers"""
        left = (self._math_object + other._math_object).left
        right = (self._math_object + other._math_object).right
        essence = self.essence
        return type(self)(essence=essence, interval_initialisation=[left, right])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """subtract two uncertain numbers"""
        left = (self._math_object - other._math_object).left
        right = (self._math_object - other._math_object).right
        essence = self.essence
        return type(self)(essence=essence, interval_initialisation=[left, right])

    def __mul__(self, other):
        """multiply two uncertain numbers"""

        if isinstance(other, int | float):
            left = (self._math_object * other).left
            right = (self._math_object * other).right
            essence = self.essence
            return type(self)(essence=essence, interval_initialisation=[left, right])
        elif isinstance(other, UncertainNumber):
            left = (self._math_object * other._math_object).left
            right = (self._math_object * other._math_object).right
            essence = self.essence
            return type(self)(essence=essence, interval_initialisation=[left, right])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """divide two uncertain numbers"""

        if isinstance(other, int | float):
            left = (self._math_object / other).left
            right = (self._math_object / other).right
            essence = self.essence
            return type(self)(essence=essence, interval_initialisation=[left, right])
        elif isinstance(other, UncertainNumber):
            left = (self._math_object / other._math_object).left
            right = (self._math_object / other._math_object).right
            essence = self.essence
            return type(self)(essence=essence, interval_initialisation=[left, right])

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __pow__(self, other):
        """power of two uncertain numbers"""

        if isinstance(other, int | float):
            left = (self._math_object**other).left
            right = (self._math_object**other).right
            essence = self.essence
            return type(self)(essence=essence, interval_initialisation=[left, right])
        elif isinstance(other, UncertainNumber):
            left = (self._math_object**other._math_object).left
            right = (self._math_object**other._math_object).right
            essence = self.essence
            return type(self)(essence=essence, interval_initialisation=[left, right])

    @classmethod
    def _toIntervalBackend(cls, vars=None):
        """transform any UN object to an `interval`

        question:
            - what is the `interval` representation: list, nd.array or Interval object?
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
                return sth.interval_initialisation
            else:
                return sth._math_object.rangel

        _UNintervals_list = [as_interval(k) for k in selected_objs]
        _UNintervals = np.array(_UNintervals_list).reshape(-1, 2)
        return _UNintervals

    @classmethod
    def _IntervaltoCompBackend(cls, vars):
        """convert the interval-tupe UNs instantiated to the computational backend

        note:
            - it will automatically convert all the UN objects in array-like to the computational backend

        returns:
            - nd.array or Marco's Interval object

        thoughts:
            - if Marco's, then we'd use `intervalise` func to get all interval objects
            and then to create another func to convert the interval objects to np.array to do endpoints method
        """

        # from augument list to intervals list
        all_objs = {instance.symbol: instance for instance in cls.instances}
        _intervals = [
            all_objs[k].interval_initialisation for k in all_objs if k in vars
        ]
        _UNintervals = np.array(_intervals).reshape(-1, 2)
        return _UNintervals

    ##### UP methods #####
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
            _UNintervals = [k.interval_initialisation for k in UN_list]
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
            interval_initialisation=(output_bounds_lo, output_bounds_hi),
            **kwargs,
        )
        # return endpoints_propagation_2n(_UNintervals, func)

    ###################################
    ##### serialisation functions #####
    ###################################

    def JSON_dump(self, filename="UN_data.json"):
        """the JSON serialisation of the UN object into the filesystem"""

        filepath = Path(Params.result_path) / filename
        with open(filepath, "w") as fp:
            json.dump(self, fp, cls=UNEncoder, indent=4)


# TODO unfinished logic: currently if suffices in creating only `Interval` object
# @classmethod
def parse_description(description):
    """Parse the description of the uncertain number

    args:
        description: str
            the flexible string desired by Scott to instantiate a Uncertain Number

    caveat:
        the description needs to have space between the values and the operators, such as '[15 +- 10%]'
    """
    # return cls(radius=diameter / 2)

    ### type 1 ###
    # initial check if string-rep of list
    if initial_list_checking(description):
        an_int = initial_list_checking(description)
        if len(an_int) == 1:
            return PM(an_int[0], hw=Params.hw)
        elif len(an_int) > 1:
            return I(*an_int)
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

    # @classmethod
    # def from_diameter(cls, type, description):
    #     """Scott desired way of instantiating

    #     note:
    #         parser from a description from the user

    #     args:
    #         type: str
    #             the type of the uncertain number, ['interval', 'pbox', 'distribution']
    #         description: str
    #             the description of the uncertain number
    #     """
    #     match type:
    #         case "interval":
    #             return "Interval type selected"
    #         case "distribution":
    #             return "Distribution type selected"
    #         case "pbox":
    #             return "Pbox type selected"
    #         case _:
    #             return "Currently only {'interval', 'distribution', and 'pbox'} types are supported"
    #     # return cls(radius=diameter / 2)

    # def __add__(self, other):
    #     """ Add two uncertain numbers.
    #     #TODO unfinished logic for adding uncertain numbers
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
