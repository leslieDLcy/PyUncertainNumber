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
from ..decorator import constructUN

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
    """Base class blueprint. Not for direct use"""

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
    """Aleatoric uncertainty propagation class for construct

    args:
        vars (Distribution): a list of uncertain numbers objects
        func (callable): the response or performance function applied to the uncertain numbers
        method (str): a string indicating the method to be used for propagation.


    note:
        Supported methods include "monte_carlo", "latin_hypercube".

    caution:
        This function supports with low-level constructs NOT the high-level `UN` (uncertain number) objects.
        For `UN` objects, use `Propagation` class as an high-level API.


        .. seealso::
            :func:`Propagation` : the high-level API for uncertain number propagation.


    example:
        >>> from pyuncertainnumber import pba
        >>> from pyuncertainnumber.propagation.p import AleatoryPropagation
        >>> def foo(x): return x[0] ** 3 + x[1] + x[2]
        >>> a = pba.Distribution('gaussian', (3,1))
        >>> b = pba.Distribution('gaussian', (10, 1))
        >>> c = pba.Distribution('uniform', (5, 10))
        >>> aleatory = AleatoryPropagation(vars=[a_d, b_d, c_d], func=foo, method='monte_carlo')
        >>> result = aleatory(n_sam=1000)
    """

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
                # regular sampling style
                try:
                    # regular sampling style
                    input_samples = [v.sample(n_sam) for v in self._vars]
                    output_samples = self.func(input_samples)
                except Exception as e:
                    # vectorised sampling style
                    input_samples = np.array(
                        [v.sample(n_sam) for v in self._vars]
                    ).T  # (n_sam, n_vars) == (n, d)
                    output_samples = self.func(input_samples)
            case "latin_hypercube" | "lhs":
                sampler = qmc.LatinHypercube(d=len(self._vars))
                lhs_samples = sampler.random(n=n_sam)  # u-space (n, d)

                try:
                    # regular sampling style
                    input_samples = [
                        v.alpha_cut(lhs_samples[:, i]) for i, v in enumerate(self._vars)
                    ]
                    output_samples = self.func(input_samples)
                except Exception as e:
                    # vectorised sampling style
                    input_samples = np.array(
                        [
                            v.alpha_cut(lhs_samples[:, i])
                            for i, v in enumerate(self._vars)
                        ]
                    ).T
                    output_samples = self.func(input_samples)
            case "taylor_expansion":
                pass
            case _:
                raise ValueError("method not yet supported")
        return output_samples


class EpistemicPropagation(P):
    """Epistemic uncertainty propagation class for construct

    args:
        vars (Interval): a list of interval objects
        func (callable): the response or performance function applied to the uncertain numbers
        method (str): a string indicating the method to be used for propagation
        interval_strategy (str): a strategy for interval propagation, including {'endpoints', 'subinterval'}

    caution:
        This function supports with low-level constructs NOT the high-level `UN` (uncertain number) objects.
        For `UN` objects, use `Propagation` class as an high-level API.


        .. seealso::
            :func:`Propagation` : the high-level API for uncertain number propagation.

    example:
        >>> from pyuncertainnumber import pba
        >>> from pyuncertainnumber.propagation.p import EpistemicPropagation
        >>> def foo(x): return x[0] ** 3 + x[1] + x[2]
        >>> a = pba.I(1, 5)
        >>> b = pba.I(7, 13)
        >>> c = pba.I(5, 10)
        >>> ep = EpistemicPropagation(vars=[a,b,c], func=foo, method='subinterval')
        >>> result = ep(n_sub=20, style='endpoints')
    """

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
    """Mixed uncertainty propagation class for construct

    args:
        vars (Pbox or DempsterShafer): a list of uncertain numbers objects
        func (callable): the response or performance function applied to the uncertain numbers
        method (str): a string indicating the method to be used for pbox propagation, including {'interval_monte_carlo', 'slicing', 'double_monte_carlo'}.
        interval_strategy (str): a strategy for interval propagation, including {'direct', 'subinterval', 'endpoints'}.

    caution:
        This function supports with low-level constructs NOT the high-level `UN` (uncertain number) objects.
        For `UN` objects, use `Propagation` class as an high-level API.


        .. seealso::
            :func:`Propagation` : the high-level API for uncertain number propagation.


    warning:
        The computation cost increases exponentially with the number of input variables and the number of slices.
        Be cautious with the choice of number of slices ``n_slices`` given the number of input variables ``vars`` of the response function.


    note:
        Discussion of the methods and strategies.
        When choosing ``interval_strategy``, "direct" requires function signature to take a list of inputs,
        whereas "subinterval" and "endpoints" require the function to take a vectorised signature.

    example:
        >>> from pyuncertainnumber import pba
        >>> from pyuncertainnumber.propagation.p import MixedPropagation
        >>> def foo(x): return x[0] ** 3 + x[1] + x[2]
        >>> a = pba.normal([2, 3], [1])
        >>> b = pba.normal([10, 14], [1])
        >>> c = pba.normal([4, 5], [1])
        >>> mix = MixedPropagation(vars=[a,b,c], func=foo, method='slicing', interval_strategy='direct')
        >>> result = mix(n_slices=20)
    """

    def __init__(self, vars, func, method, interval_strategy=None):

        super().__init__(vars, func, method)
        self.interval_strategy = interval_strategy
        self.post_init_check()

    # assume striped UN classes (i.e. constructs only)
    def type_check(self):
        """Inspection if inputs are mixed uncertainy model"""

        has_I = any(isinstance(item, Interval) for item in self._vars)
        has_D = any(isinstance(item, Distribution) for item in self._vars)
        has_P = any(isinstance(item, Pbox) for item in self._vars)

        assert (has_I and has_D) or has_P, "Not a mixed uncertainty problem"

    def method_check(self):
        """Check if the method is suitable for mixed uncertainty propagation"""
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
            case None:
                handler = slicing
            case _:
                raise ValueError("Unknown method")

        results = handler(self._vars, self.func, self.interval_strategy, **kwargs)
        return results


# * ------------------ Uncertain Number Propagation ------------------ *
class Propagation:
    """High-level integrated class for the propagation of uncertain numbers

    args:
        vars (UncertainNumber): a list of uncertain numbers objects
        func (Callable): the response or performance function applied to the uncertain numbers
        method (str):
            a string indicating the method to be used for propagation (e.g. "monte_carlo", "endpoint", etc.) which may depend on the constructs of the uncertain numbers.
            See notes about function signature.
        interval_strategy (str):
            a strategy for interval propagation, including {'direct', 'subinterval', 'endpoints'} which will
            affect the function signature of the response function. See notes about function signature.

    caution:
        This class supports with high-level computation with `UncertainNumber` objects.

    note:
        Discussion of the methods and strategies.
        When choosing ``interval_strategy``, "direct" requires function signature to take a list of inputs,
        whereas "subinterval" and "endpoints" require the function to take a vectorised signature.

    warning:
        The computation cost increases exponentially with the number of input variables and the number of slices.
        Be cautious with the choice of number of slices ``n_slices`` given the number of input variables ``vars`` of the response function.

    example:
        >>> import pyuncertainnumber as pun
        >>> # construction of uncertain number objects
        >>> a = pun.I(2, 3)
        >>> b = pun.normal(4, 1)
        >>> c = pun.uniform([4,5], [9,10])

        >>> # vectorised function signature with matrix input (2D np.ndarray)
        >>> def foo_vec(x): return x[:, 0] ** 3 + x[:, 1] + x[:, 2]

        >>> # high-level propagation API
        >>> p = Propagation(vars=[a,b,c],
        >>>     func=foo,
        >>>     method='slicing',
        >>>     interval_strategy='subinterval'
        >>> )

        >>> # heavy-lifting of propagation
        >>> t = p.run(n_sam=20, n_sub=2, style='endpoints')
    """

    def __init__(
        self,
        vars: list[UncertainNumber],
        func: callable,
        method: str,
        interval_strategy: str = None,
    ):

        self._vars = vars
        self._func = func
        self.method = method
        self.interval_strategy = interval_strategy
        self._post_init_check()

    def _post_init_check(self):
        """Some checks after initialisation"""

        # strip the underlying constructs from UN
        self._constructs = [c._construct for c in self._vars]

        # supported methods check

        # assign method herein
        self.assign_method()

    def assign_method(self):
        """Assign the propagation method based on the essence of constructs"""
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

    @constructUN
    def run(self, **kwargs):
        """Doing the propagation and return UN

        return:
            UncertainNumber: the result of the propagation as an uncertain number object
        """

        # choose the method accordingly
        return self.p(**kwargs)
