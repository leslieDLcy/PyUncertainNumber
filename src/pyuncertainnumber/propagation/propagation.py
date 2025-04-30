from __future__ import annotations
from typing import TYPE_CHECKING

from .epistemic_uncertainty.endpoints import endpoints_method
from .epistemic_uncertainty.extremepoints import extremepoints_method
from .epistemic_uncertainty.subinterval import subinterval_method
from .epistemic_uncertainty.sampling import sampling_method
from .epistemic_uncertainty.genetic_optimisation import genetic_optimisation_method
from .epistemic_uncertainty.local_optimisation import local_optimisation_method
from .epistemic_uncertainty.endpoints_cauchy import cauchydeviates_method
from .aleatory_uncertainty.sampling_aleatory import sampling_aleatory_method
from .mixed_uncertainty.mixed_up import (
    interval_monte_carlo,
    slicing,
    double_monte_carlo,
)
from ..pba.intervalOperators import make_vec_interval

"""the new top-level module for the propagation of uncertain numbers"""

"""crossover logic

UncertainNumber: ops are indeed the ops for the underlying constructs

"""


if TYPE_CHECKING:
    from ..characterisation.uncertainNumber import UncertainNumber


from abc import ABC, abstractmethod


class P(ABC):
    def __init__(self, vars, func, method, save_raw_data: bool = False):
        self.vars = vars
        self.func = func
        self.method = method
        self.save_raw_data = save_raw_data

    def _post_init(self):
        """some checks"""

        assert callable(self.func), "function is not callable"
        self.type_check()

    @abstractmethod
    def type_check(self):
        """if the nature of the UN suitable for the method"""
        pass

    @abstractmethod
    def method_check(self):
        """if the method is suitable for the nature of the UN"""
        pass


class AleatoryPropagation(P):

    from .aleatory_uncertainty.sampling_aleatory import sampling_aleatory_method

    def __init__(self, vars, func, method, save_raw_data: bool = False):
        super().__init__(vars, func, method, save_raw_data)

    def type_check(self):
        """only distributions"""
        from ..pba.distributions import Distribution

        assert all(isinstance(v, Distribution) for v in self.vars) or all(
            isinstance(v.construct, Distribution) for v in self.vars
        ), "Not all variables are distributions"

    def method_check(self):
        assert self.method in [
            "monte_carlo",
            "latin_hypercube",
        ], "Method not supported for aleatory uncertainty propagation"

    def __call__(self):
        """doing the propagation"""
        pass


class EpistemicPropagation(P):
    def __init__(self, vars, func, method, save_raw_data: bool = False):
        super().__init__(vars, func, method, save_raw_data)

    def type_check(self):
        """only intervals"""

        from ..pba.intervals.number import Interval

        assert all(isinstance(v.construct, Interval) for v in self.vars) or all(
            isinstance(v, Interval) for v in self.vars
        ), "Not all variables are intervals"

    def method_check(self):
        assert self.method in [
            "endpoint",
            "endpoints",
            "vertex",
            "extremepoints",
            "subinterval",
            "subinterval_reconstitution",
            "cauchy",
            "endpoint_cauchy",
            "endpoints_cauchy",
            "local_optimisation",
            "local_optimization",
            "local optimisation",
            "genetic_optimisation",
            "genetic_optimization",
            "genetic optimization",
            "genetic optimisation",
        ], f"Method {self.method} not supported for epistemic uncertainty propagation"

    def __call__(self, **kwargs):
        #! caveat: possibly requires more kwargs for some methods
        """doing the propagation"""
        match self.method:
            case "endpoint" | "endpoints" | "vertex":
                handler = endpoints_method
            case "extremepoints":
                handler = extremepoints_method
            case "subinterval" | "subinterval_reconstitution":
                handler = subinterval_method
            case "cauchy" | "endpoint_cauchy" | "endpoints_cauchy":
                handler = cauchydeviates_method
            case (
                "local_optimization"
                | "local_optimisation"
                | "local optimisation"
                | "local optimization"
            ):
                handler = local_optimisation_method
            case (
                "genetic_optimisation"
                | "genetic_optimization"
                | "genetic optimization"
                | "genetic optimisation"
            ):
                handler = genetic_optimisation_method
            case _:
                raise ValueError("Unknown method")

        # TODO: make the methods signature consistent
        # TODO: ONLY an response interval needed to be returned
        results = handler(
            make_vec_interval(self.vars),
            self.func,
            self.save_raw_data,
            **kwargs,
        )
        return results


class MixedPropagation(P):
    def __init__(self, vars, func, method, save_raw_data: bool = False):
        super().__init__(vars, func, method, save_raw_data)

    # assume striped UM classes
    def type_check(self):
        """mixed UM"""
        from ..pba.pbox_abc import Box
        from ..pba.intervals.number import Interval
        from ..pba.distributions import Distribution

        has_I = any(isinstance(item, Interval) for item in self.vars)
        has_D = any(isinstance(item, Distribution) for item in self.vars)
        has_P = any(isinstance(item, Box) for item in self.vars)

        assert (has_I and has_D) or has_P, "Not a mixed uncertainty problem"

    def method_check(self):
        assert self.method in [
            "interval_monte_carlo",
            "slicing",
            "double_monte_carlo",
        ], f"Method {self.method} not supported for mixed uncertainty propagation"

    def __call__(self, **kwargs):
        """doing the propagation"""
        match self.method:
            case "interval_monte_carlo":
                handler = interval_monte_carlo
            case "slicing":
                handler = slicing
            case "double_monte_carlo":
                handler = double_monte_carlo
            case _:
                raise ValueError("Unknown method")

        results = handler(
            self.vars, self.func, self.method, self.save_raw_data, **kwargs
        )
        return results


# top-level
class Propagation:
    # TODO I'd like to strip UN classes into construct classes herein
    def __init__(
        self,
        vars: list[UncertainNumber],
        function: callable,
        method,
        save_raw_data: bool = False,
        **kwargs,
    ):
        self._vars = vars
        self._func = function
        self.method = method
        self.save_raw_data = save_raw_data

    def _post_init_check(self):

        # supported methods check
        pass

    def __call__(self, **kwargs):
        """doing the propagation"""
        pass
