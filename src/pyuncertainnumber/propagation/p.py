from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial
from .epistemic_uncertainty.extremepoints import extremepoints_method
from .epistemic_uncertainty.genetic_optimisation import genetic_optimisation_method
from .epistemic_uncertainty.local_optimisation import local_optimisation_method
from .epistemic_uncertainty.endpoints_cauchy import cauchydeviates_method
from .mixed_uncertainty.mixed_up import (
    interval_monte_carlo,
    slicing,
    double_monte_carlo,
)
from ..pba.intervals.intervalOperators import make_vec_interval
import numpy as np
from scipy.stats import qmc

from abc import ABC, abstractmethod
from ..pba.pbox_abc import Pbox
from ..pba.intervals.number import Interval
from ..pba.distributions import Distribution
from ..propagation.epistemic_uncertainty.b2b import b2b

"""the new top-level module for the propagation of uncertain numbers"""

"""crossover logic

UncertainNumber: ops are indeed the ops for the underlying constructs

"""


if TYPE_CHECKING:
    from ..characterisation.uncertainNumber import UncertainNumber


import logging

# Basic configuration for logging
logging.basicConfig(level=logging.INFO)


# * ------------------ constructs Propagation ------------------ *
class P(ABC):
    def __init__(self, vars, func, method, save_raw_data: bool = False):
        self._vars = vars
        self.func = func
        self.method = method
        self.save_raw_data = save_raw_data

    def post_init_check(self):
        """some checks"""

        assert callable(self.func), "function is not callable"
        self.type_check()
        self.method_check()

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
        self.post_init_check()

    def type_check(self):
        """only distributions"""
        from ..pba.distributions import Distribution
        from ..pba.pbox_abc import Pbox

        assert all(
            isinstance(v, Distribution | Pbox) for v in self._vars
        ), "Not all variables are distributions"

    def method_check(self):
        assert self.method in [
            "monte_carlo",
            "latin_hypercube",
        ], "Method not supported for aleatory uncertainty propagation"

    def __call__(self, n_sam: int = 1000):
        """doing the propagation"""
        match self.method:
            case "monte_carlo":
                input_samples = np.array(
                    [v.sample(n_sam) for v in self._vars]
                ).T  # (n_sam, n_vars) == (n, d)
                output_samples = self.func(input_samples)
            case "latin_hypercube" | "lhs":
                sampler = qmc.LatinHypercube(d=len(self._vars))
                lhs_samples = sampler.random(n=n_sam)  # u-space (n, d)
                input_samples = np.array(
                    [v.alpha_cut(lhs_samples[:, i]) for i, v in enumerate(self._vars)]
                ).T
                # ! a shape check
                # print("shape check of input samples", input_samples.shape)
                output_samples = self.func(input_samples)
            case "taylor_expansion":
                pass
            case _:
                raise ValueError("method not yet supported")
        return output_samples


class EpistemicPropagation(P):
    def __init__(self, vars, func, method):
        super().__init__(vars, func, method)
        self.post_init_check()

    def type_check(self):
        """only intervals"""

        from ..pba.intervals.number import Interval

        assert all(
            isinstance(v, Interval) for v in self._vars
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
        ], f"Method {self.method} not supported for epistemic uncertainty propagation"

    def __call__(self, **kwargs):
        #! caveat: possibly requires more kwargs for some methods
        """doing the propagation"""
        match self.method:
            case "endpoint" | "endpoints" | "vertex":
                handler = partial(b2b, interval_strategy="endpoints")
            case "extremepoints":
                handler = extremepoints_method
            case "subinterval" | "subinterval_reconstitution":
                handler = partial(b2b, interval_strategy="subinterval")
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
            make_vec_interval(self._vars),  # pass down vec interval
            self.func,
            **kwargs,
        )
        return results


class MixedPropagation(P):
    def __init__(self, vars, func, method, interval_strategy=None):
        """initialisation

        args:
            interval_strategy: a certain strategy used for the interval propagation
                such as endpoints, subinterval, etc. By default, it is set to None
        """
        super().__init__(vars, func, method)
        self.interval_strategy = interval_strategy
        self.post_init_check()

    # assume striped UM classes
    def type_check(self):
        """mixed UM"""

        has_I = any(isinstance(item, Interval) for item in self._vars)
        has_D = any(isinstance(item, Distribution) for item in self._vars)
        has_P = any(isinstance(item, Pbox) for item in self._vars)

        # assert (has_I and has_D) or has_P, "Not a mixed uncertainty problem"

    def method_check(self):
        assert self.method in [
            "interval_monte_carlo",
            "slicing",
            "equi_cutting",
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
            case None:
                handler = slicing
            case _:
                raise ValueError("Unknown method")

        results = handler(self._vars, self.func, self.interval_strategy, **kwargs)
        return results


# * ------------------ Uncertain Number Propagation ------------------ *
class Propagation:
    """high-level integrated class for the propagation of uncertain numbers

    args:
        vars: a list of uncertain numbers objects
        func: the response or performance function applied to the uncertain numbers
        method: a string indicating the method to be used for propagation (e.g. "monte_carlo", "endpoint", etc.)
        interval_strategy: a strategy for interval propagation, if applicable (e.g. subinterval, etc.)
    """

    def __init__(
        self,
        vars: list[UncertainNumber],
        func: callable,
        method,
        interval_strategy=None,
    ):
        """top-level class for the propagation of uncertain numbers

        args:
            vars: a list of uncertain numbers objects
        """
        self._vars = vars
        self._func = func
        self.method = method
        self.interval_strategy = interval_strategy
        self._post_init_check()

    def _post_init_check(self):

        # strip the underlying constructs from UN
        self._constructs = [c._construct for c in self._vars]

        # supported methods check

        # assign method herein
        self.assign_method()

    def assign_method(self):
        # created an underlying propagation `self.p` object

        # all
        all_I = all(isinstance(item, Interval) for item in self._constructs)
        all_D = all(isinstance(item, Distribution) for item in self._constructs)
        # any
        has_I = any(isinstance(item, Interval) for item in self._constructs)
        has_D = any(isinstance(item, Distribution) for item in self._constructs)
        has_P = any(isinstance(item, Pbox) for item in self._constructs)

        if all_I:
            # all intervals
            logging.info("interval propagation")
            self.p = EpistemicPropagation(self._constructs, self._func, self.method)
        elif all_D:
            logging.info("distribution propagation")
            # all distributions
            self.p = AleatoryPropagation(self._constructs, self._func, self.method)
        elif (has_I and has_D) or has_P:
            # mixed uncertainty
            logging.info("mixed uncertainty propagation")
            self.p = MixedPropagation(
                self._constructs,
                self._func,
                self.method,
                self.interval_strategy,
                # interval_strategy=self.kwargs.get("interval_strategy", None),
            )
        else:
            raise ValueError(
                "Not a valid combination of uncertainty types. "
                "Please check the input variables."
            )

    @property
    def constructs(self):
        """return the underlying constructs"""
        return self._constructs

    def run(self, **kwargs):
        """doing the propagation"""

        # choose the method accordingly
        return self.p(**kwargs)
