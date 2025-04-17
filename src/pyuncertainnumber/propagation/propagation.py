"""the new top-level module for the propagation of uncertain numbers"""

"""crossover logic

UncertainNumber: ops are indeed the ops for the underlying constructs

"""


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


if TYPE_CHECKING:
    from ..characterisation.uncertainNumber import UncertainNumber


from abc import ABC, abstractmethod


class P(ABC):
    def __init__(self, vars, func, method, save_results: bool = False):
        self.vars = vars
        self.func = func
        self.method = method
        self.save_results = save_results

    def _post_init(self):
        """some checks"""

        assert callable(self.func), "function is not callable"
        self.type_check()

    @abstractmethod
    def type_check(self):
        """if the nature of the UN suitable for the method"""
        pass


class AleatoryPropagation(P):

    from .aleatory_uncertainty.sampling_aleatory import sampling_aleatory_method

    def __init__(self, vars, func, method, save_results: bool = False):
        super().__init__(vars, func, method, save_results)

    def type_check(self):
        """only distributions"""
        from ..pba.distributions import Distribution

        assert all(
            [isinstance(v, Distribution) for v in self.vars]
        ), "Not all variables are distributions"

    def __call__(self):
        """doing the propagation"""
        pass


class EpistemicPropagation(P):
    def __init__(self, vars, func, method, save_results: bool = False):
        super().__init__(vars, func, method, save_results)

    def type_check(self):
        """only intervals"""

        from ..pba.intervals.number import Interval

        assert all(
            [isinstance(v, Interval) for v in self.vars]
        ), "Not all variables are intervals"

    def __call__(self, **kwargs):
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

        results = handler(
            self.vars, self.func, self.method, self._save_results, **kwargs
        )
        return results


class Propagation:

    def __init__(
        self,
        vars: list[UncertainNumber],
        function: callable,
        method,
        save_results: bool = False,
        **kwargs,
    ):
        self._vars = vars
        self._func = function
        self.method = method
        self._save_results = save_results

    def _post_init_check(self):

        # supported methods check
        pass

    def __call__(self, **kwargs):
        """doing the propagation"""
        pass
