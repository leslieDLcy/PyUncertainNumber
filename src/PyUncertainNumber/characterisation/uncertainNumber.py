from dataclasses import dataclass, field
from typing import Type, Union, List
import functools

# from .measurand import Measurand
# from .variability import Variability
from .uncertainty_types import Uncertainty_types
from .ensemble import Ensemble
from ..pba.interval import Interval as nInterval
from .utils import *
from ..pba.params import Params
from pint import UnitRegistry
from pathlib import Path
import itertools
from ..nlp.language_parsing import hedge_interpret
from scipy.stats import norm
from .check import DistributionSpecification
from ..pba.pbox import named_pbox
from typing import Sequence
from ..pba.distributions import Distribution, named_dists
from ..pba.operation import convert
from ..pba.intervalOperators import parse_bounds

""" Uncertain Number class """


@dataclass
class UncertainNumber:
    """Uncertain Number class

    args:
        - `bounds`;
        - `distribution_parameters`: a list of the distribution family and its parameters; e.g. ['norm', [0, 1]];
        - `pbox_initialisation`: a list of the distribution family and its parameters; e.g. ['norm', ([0,1], [3,4])];
        -  naked_value: the deterministic numeric representation of the UN object, which shall be linked with the 'pba' or `Intervals` package

    Example:
        >>> UncertainNumber(name="velocity", symbol="v", units="m/s", bounds=[1, 2])
    """

    # ---------------------Basic---------------------#
    name: str = field(default=None)
    symbol: str = field(default=None)
    # string input of units, e.g. 'm/s'
    units: Type[any] = field(default=None, repr=False)
    _Q: Type[any] = field(default=None, repr=False)

    # ---------------------Value---------------------#
    # ensemble: Type[Ensemble] = field(default=None)
    uncertainty_type: Type[Uncertainty_types] = field(default=None, repr=False)
    essence: str = field(default=None)  # [interval, distribution, pbox, ds]
    masses: list[float] = field(default=None, repr=False)
    bounds: Union[List[float], str] = field(default=None)
    distribution_parameters: list[str, float | int] = field(default=None)
    pbox_parameters: list[str, Sequence[nInterval]] = field(default=None, repr=False)
    hedge: str = field(default=None, repr=False)
    _construct: Type[any] = field(default=None, repr=False)
    # this is the deterministic numeric representation of the
    # UN object, which shall be linked with the 'pba' or `Intervals` package
    naked_value: float = field(default=None)
    p_flag: bool = field(default=True, repr=False)  # parameterised flag

    # * ---------------------auxlliary information---------------------*#
    # some simple boiler plates
    # lat: float = field(default=0.0, metadata={'unit': 'degrees'})
    # ensemble: Type[Ensemble] = field(default=None)

    measurand: str = field(default=None, repr=False)
    nature: str = field(default=None, repr=False)
    provenence: str = field(default=None, repr=False)
    justification: str = field(default=None, repr=False)
    structure: str = field(default=None, repr=False)
    security: str = field(default=None, repr=False)

    # * ---------------------aleatoric component--------------------- *#
    ensemble: Type[Ensemble] = field(default=None, repr=False)
    variability: str = field(default=None, repr=False)
    dependence: str = field(default=None, repr=False)

    # * ---------------------epistemic component---------------------*#
    uncertainty: str = field(default=None, repr=False)

    # class variable
    instances = []  # TODO named as registry later on

    # * --------------------- additional ---------------------*#
    _samples: np.ndarray | list = field(default=None, repr=False)

    # *  ---------------------more on initialisation---------------------*#
    def parameterised_pbox_specification(self):
        if self.p_flag:
            self._construct = self.match_pbox(
                self.distribution_parameters[0],
                self.distribution_parameters[1],
            )
            self.naked_value = self._construct.mean().midpoint()

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
            if (self._construct is None) | (not self._construct):
                print("a vacuous interval is created")
                self.essence = "interval"
                self.bounds = [-np.inf, np.inf]

        UncertainNumber.instances.append(self)

        ### create the underlying construct ###
        match self.essence:
            case "interval":
                self._construct = parse_bounds(self.bounds)
                self.naked_value = self._construct.midpoint()
            case "distribution":
                if self._samples is not None:
                    self._construct = Distribution(sample_data=self._samples)
                elif self.distribution_parameters is not None:
                    p_ = DistributionSpecification(
                        dist_family=self.distribution_parameters[0],
                        dist_params=self.distribution_parameters[1],
                    )
                    if p_.i_flag:
                        self.parameterised_pbox_specification()
                    else:
                        self._construct = Distribution(
                            dist_family=self.distribution_parameters[0],
                            dist_params=self.distribution_parameters[1],
                        )
                        self.naked_value = self._construct._naked_value
            case "pbox":
                self.parameterised_pbox_specification()

        ### 'unit' representation of the un ###
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        # I can use the following logic to double check the arithmetic operations of the UN object
        if isinstance(self.naked_value, float | int):
            self._Q = Q_(self.naked_value, self.units)
            # self.naked_value * ureg(self.units)  # Quantity object
        else:
            self._Q = Q_(1, self.units)

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
        field_values = {k: v for k, v in self.__dict__.items() if v is not None}
        field_str = ", ".join(f"{k}={repr(v)}" for k, v in field_values.items())
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

    # * ---------------------getters --------------------- *#
    @property
    def construct(self):
        return self._construct

    # * ---------------------other constructors--------------------- *#

    # @classmethod
    # def I(cls, i: list[float | int]):
    #     """create a shorcut an interval-type UN object"""
    #     return cls(essence="interval", bounds=i)

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
    def fromConstruct(cls, construct):
        """create an Uncertain Number from a construct object"""
        from ..pba.pbox_base import Pbox
        from ..pba.interval import Interval as nInterval
        from ..pba.ds import DempsterShafer
        from ..pba.distributions import Distribution

        if isinstance(construct, Pbox):
            return cls.from_pbox(construct)
        if isinstance(construct, nInterval):
            return cls.from_Interval(construct)
        if isinstance(construct, DempsterShafer):
            return cls.from_ds(construct)
        if isinstance(construct, Distribution):
            return cls.fromDistribution(construct)
        else:
            raise ValueError("The construct object is not recognised")

    @classmethod
    def fromDistribution(cls, D, **kwargs):
        # dist_family: str, dist_params,
        """create an Uncertain Number from specification of distribution

        args:
            - D: Distribution object
            dist_family (str): the distribution family
            dist_params (list, tuple or string): the distribution parameters
        """
        distSpec = DistributionSpecification(D.dist_family, D.dist_params)

        if D.sample_data is None:
            return cls(
                essence="distribution",
                distribution_parameters=distSpec.get_specification(),
                **kwargs,
            )
        else:
            return cls(
                essence="distribution",
                distribution_parameters=None,
                _samples=D.sample_data,
            )

    # @classmethod
    # def from_constraints(cls, min, max, mean, median, variance, **kwargs):
    #     """to construct a pbox given the properties of the distribution

    #     returns:
    #         - a pbox-type UN object
    #     note:
    #         - whether differentiate explicitly if free/parametric pbox
    #     """
    #     pass
    @classmethod
    def from_Interval(cls, u):
        return cls(essence="interval", bounds=u)

    @classmethod
    def from_pbox(cls, p):
        """genenal from  pbox"""
        # passPboxParameters()
        return cls(essence="pbox", p_flag=False, _construct=p)

    @classmethod
    def from_ds(cls, ds):
        cls.from_pbox(ds.to_pbox())

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

    # * ---------------------unary operations---------------------#
    def sqrt(self):
        return self._construct.sqrt()

    # * ---------------------binary operations---------------------#

    def __add__(self, other):
        """add two uncertain numbers"""
        if isinstance(other, float | int | np.number):
            other = UncertainNumber.from_Interval(nInterval(other))
        a, b = self._construct, other._construct
        if isinstance(a, nInterval) and isinstance(b, nInterval):
            r = a + b
            return UncertainNumber.from_Interval(r)
        else:
            r = convert(a) + convert(b)
        # TODO unit handling for arithmetic operations not implemented
        # TODO due to the stupid multi Registry error
        # newQ = self._Q + other._Q
        return UncertainNumber.from_pbox(r)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        if isinstance(other, float | int | np.number):
            other = UncertainNumber.from_Interval(nInterval(other))
        a, b = self._construct, other._construct

        if isinstance(a, nInterval) and isinstance(b, nInterval):
            r = a - b
            return UncertainNumber.from_Interval(r)
        else:
            r = convert(a) - convert(b)
        return UncertainNumber.from_pbox(r)

    def __mul__(self, other):
        """multiply two uncertain numbers"""

        if isinstance(other, float | int | np.number):
            other = UncertainNumber.from_Interval(nInterval(other))
        a, b = self._construct, other._construct
        if isinstance(a, nInterval) and isinstance(b, nInterval):
            r = a * b
            return UncertainNumber.from_Interval(r)
        else:
            r = convert(a) * convert(b)
        return UncertainNumber.from_pbox(r)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """divide two uncertain numbers"""

        if isinstance(other, float | int | np.number):
            other = UncertainNumber.from_Interval(nInterval(other))

        a, b = self._construct, other._construct
        if isinstance(a, nInterval) and isinstance(b, nInterval):
            r = a / b
            return UncertainNumber.from_Interval(r)
        else:
            r = convert(a) / convert(b)
        return UncertainNumber.from_pbox(r)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __pow__(self, other):
        """power of two uncertain numbers"""

        if isinstance(other, float | int | np.number):
            other = UncertainNumber.from_Interval(nInterval(other))
        a, b = self._construct, other._construct
        if isinstance(a, nInterval) and isinstance(b, nInterval):
            r = a**b
            return UncertainNumber.from_Interval(r)
        else:
            r = convert(a) ** convert(b)
        return UncertainNumber.from_pbox(r)

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
            - essentially vars shall be all interval-typed UNs by now

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

    # @classmethod
    # def vertexMethod(cls, vars, func):
    #     """implementation of the endpoints method for the uncertain number

    #     args:
    #         vars: list
    #             the selected list of the symbols of UN or a list of arrays
    #         func: function
    #             the function to be applied to the uncertain number
    #     """

    #     if isinstance(vars[0], str):
    #         # _UNintervals = UncertainNumber._IntervaltoCompBackend(vars) # bp
    #         _UNintervals = UncertainNumber._toIntervalBackend(vars)
    #         df = vM(_UNintervals, func)
    #         return df
    #     elif isinstance(vars[0], int | float):
    #         # create a list of UN objects using hedge interpretation
    #         def get_hedgedUN(a_num_list):
    #             return [cls.from_hedge(f"{i}") for i in a_num_list]

    #         UN_list = get_hedgedUN(vars)
    #         _UNintervals = [k.bounds for k in UN_list]
    #         _UNintervals = np.array(_UNintervals).reshape(-1, 2)

    #         df = vM(_UNintervals, func)

    #         return df

    # @classmethod
    # def endpointsMethod(cls, vars, func, **kwargs):
    #     """implementation of the endpoints method for the uncertain number using
    #     Marco's implementation

    #     note:
    #         `vars` shall be consistent with the signature of `func`. This means that
    #         only a selected list of uncertain numbers will be used according to the func provided.

    #     args:
    #         vars: list
    #             the chosen list of uncertain numbers
    #         func: function
    #             the function to be applied to the uncertain number
    #     """
    #     # _UNintervals = UncertainNumber._IntervaltoCompBackend(vars) # bp
    #     _UNintervals = UncertainNumber._toIntervalBackend(vars)
    #     output_bounds_lo, output_bounds_hi, _, _ = endpoints_propagation_2n(
    #         _UNintervals, func
    #     )
    #     return cls(
    #         essence="interval",
    #         bounds=(output_bounds_lo, output_bounds_hi),
    #         **kwargs,
    #     )
    #     # return endpoints_propagation_2n(_UNintervals, func)

    # ---------------------serialisation functions---------------------#

    def JSON_dump(self, filename="UN_data.json"):
        """the JSON serialisation of the UN object into the filesystem"""

        filepath = Path(Params.result_path) / filename
        with open(filepath, "w") as fp:
            json.dump(self, fp, cls=UNEncoder, indent=4)

    def random(self, size=None):
        """Generate random samples from the distribution."""
        match self.essence:
            case "interval":
                return ValueError(
                    "Random sampling is only supported for distribution-type UncertainNumbers."
                )
            case "distribution":
                which_dist = self.distribution_parameters[0]
                return named_dists[which_dist].rvs(
                    *self.distribution_parameters[1], size=size
                )
            case "pbox":
                return ValueError(
                    "Random sampling is only supported for distribution-type UncertainNumbers."
                )

    def ppf(self, q=None):
        """ "Calculate the percent point function (inverse of CDF) at quantile q."""
        match self.essence:
            case "interval":
                return ValueError(
                    "PPF calculation is not supported for interval-type UncertainNumbers."
                )
            case "distribution":
                which_dist = self.distribution_parameters[0]
                # Define the distribution
                dist = named_dists[which_dist](*self.distribution_parameters[1])
                return dist.ppf(q)
            case "pbox":
                which_dist = self.distribution_parameters[0]
                # Assuming the p-box parameters are stored as a list of intervals
                param_specs = self.distribution_parameters[1]

                # Helper function to create parameter lists from specs
                def get_param_list(spec):
                    if isinstance(spec, list):
                        return spec  # Already a list
                    else:
                        # Create a list with repeated value
                        return [spec, spec]

                # Generate parameter lists
                param_lists = [get_param_list(spec) for spec in param_specs]

                # Generate all combinations of lower and upper bounds for each parameter
                param_combinations = list(itertools.product(*param_lists))

                # Calculate the PPF for each combination (handling array q)
                if isinstance(q, np.ndarray):
                    ppf_values = np.array(
                        [
                            [named_dists[which_dist](*params).ppf(qi) for qi in q]
                            for params in param_combinations
                        ]
                    )
                    return np.array(
                        [np.min(ppf_values, axis=0), np.max(ppf_values, axis=0)]
                    )

                else:  # If q is a single value
                    ppf_values = []
                    for params in param_combinations:
                        dist = named_dists[which_dist](*params)
                        ppf_value = dist.ppf(q)

                        if isinstance(ppf_value, np.ndarray):
                            ppf_values.extend(ppf_value)
                        else:
                            ppf_values.append(ppf_value)

                    return [min(ppf_values), max(ppf_values)]


# * ---------------------shortcuts --------------------- *#
def makeUNPbox(func):

    from ..pba.pbox import _bound_pcdf
    from ..pba.intervalOperators import wc_interval

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        i_args = [wc_interval(arg) for arg in args]
        shape_value = func(*args, **kwargs)
        p = _bound_pcdf(shape_value, *i_args)
        return UncertainNumber.fromConstruct(p)

    return wrapper_decorator


def I(i: str | list[float | int]) -> UncertainNumber:
    """a shortcut for the interval-type UN object"""
    return UncertainNumber.fromConstruct(parse_bounds(i))


@makeUNPbox
def norm(*args):
    return "norm"


@makeUNPbox
def expon(*args):
    return "expon"


@makeUNPbox
def gamma(*args):
    return "gamma"


# * ---------------------parse inputs for UN only  --------------------- *#


def _parse_interverl_inputs(vars):
    """Parse the input intervals

    note:
        - Ioanna's funcs typically take 2D NumPy arra
    """

    if isinstance(vars, np.ndarray):
        if vars.shape[1] != 2:
            raise ValueError(
                "vars must be a 2D array with two columns per row (lower and upper bounds)"
            )
        else:
            return vars

    if isinstance(vars, list):
        return UncertainNumber._toIntervalBackend(vars)
