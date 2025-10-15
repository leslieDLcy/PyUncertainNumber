from functools import partial
from abc import ABC, abstractmethod
from pyuncertainnumber import pba
from ..characterisation.uncertainNumber import UncertainNumber
from ..pba.pbox_abc import Pbox
from ..pba.intervals.number import Interval
from ..pba.distributions import Distribution
from ..pba.dependency import Dependency
from .b2b import b2b
from ..decorator import constructUN
from .local_optimisation import local_optimisation_method
from .mixed_up import (
    interval_monte_carlo,
    slicing,
    double_monte_carlo,
)
from ..pba.intervals.intervalOperators import make_vec_interval
import numpy as np


"""the new top-level module for the propagation of uncertain numbers"""

import logging

logging.basicConfig(level=logging.INFO)


# * ------------------ constructs Propagation ------------------ *
class P(ABC):
    """Base class blueprint. Not for direct use"""

    def __init__(self, vars, func, method, dependency=None):
        self._vars = vars
        self.func = func
        self.method = method
        self.dependency = dependency

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
    """Aleatoric uncertainty propagation class for Distribution constructs only

    args:
        vars (Distribution): a list of uncertain numbers objects

        func (callable): the response or performance function applied to the uncertain numbers

        method (str): a string indicating the method to be used for propagation.

        dependency (string or Dependency): a Dependency object(i.e. a copula function) to model the dependency structure among input variables.
            Strings such as "independence" accepted for independence.


    note:
        Supported methods include "monte_carlo". Note that "taylor_expansion" is not supported herein but implemented as a standalone function in the module `taylor_expansion.py`.

    caution:
        This function supports with low-level constructs NOT the high-level `UN` (uncertain number) objects.
        For `UN` objects, use `Propagation` class as an high-level API.


        .. seealso::
            :func:`Propagation` : the high-level API for uncertain number propagation.


    example:
        >>> from pyuncertainnumber import pba
        >>> from pyuncertainnumber.propagation.p import AleatoryPropagation
        >>> def foo(x): return x[0] ** 3 + x[1] + x[2]
        >>> a_d = pba.Distribution('gaussian', (3,1))
        >>> b_d = pba.Distribution('gaussian', (10, 1))
        >>> c_d = pba.Distribution('uniform', (5, 10))
        >>> aleatory = AleatoryPropagation(vars=[a_d, b_d, c_d], func=foo, method='monte_carlo')
        >>> result = aleatory(n_sam=1000)
    """

    from .taylor_expansion import taylor_expansion_method

    def __init__(self, vars, func, method, dependency=None):
        super().__init__(vars, func, method, dependency)
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

    def run(self, n_sam: int = 1000):
        """doing the propagation"""
        match self.method:
            case "monte_carlo":
                if self.dependency is None or self.dependency == "independence":
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
                        return output_samples
                else:
                    j = pba.JointDistribution(
                        copula=self.dependency, marginals=self._vars
                    )

                    s_inputs = j.sample(n_sam)  # (n_sam, n_vars)

                    output_samples = self.func(s_inputs)
                    return output_samples
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
        pass

    def run(self, **kwargs):
        #! caveat: possibly requires more kwargs for some methods
        """doing the propagation"""
        match self.method:
            case "endpoint" | "endpoints" | "vertex":
                handler = partial(b2b, interval_strategy="endpoints")
            case "subinterval" | "subintervals" | "subinterval_reconstitution":
                handler = partial(b2b, interval_strategy="subinterval")
            case (
                "cauchy"
                | "cauchy_deviate_method"
                | "endpoint_cauchy"
                | "endpoints_cauchy"
            ):
                handler = partial(b2b, interval_strategy="cauchy_deviate")
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
                handler = partial(b2b, interval_strategy="ga")
            case "bayesian_optimisation" | "bo":
                handler = partial(b2b, interval_strategy="bo")
            case _:
                raise ValueError("Unknown method")

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

        dependency (string or Dependency): a Dependency object(i.e. a copula function) to model the dependency structure among input variables.
            Strings such as "independence" accepted for independence.

        interval_strategy (str): a sub-level strategy selector for interval propagation, including {'direct', 'subinterval', 'endpoints'}.

    caution:
        This function supports with low-level constructs NOT the high-level `UN` (uncertain number) objects.
        For `UN` objects, use `Propagation` class as an high-level API.


        .. seealso::
            :func:`Propagation` : the high-level API for uncertain number propagation.


    warning:
        The computation cost increases exponentially with the number of input variables and the number of slices.
        Be cautious with the choice of number of slices ``n_slices`` given the number of input variables ``vars`` of the response function.


    note:
        Discussion of the methods and strategies. When choosing ``interval_strategy``, "direct" requires function signature to take a list of inputs,
        whereas "subinterval" and "endpoints" require the function to take a vectorised signature. Currently, only "interval_monte_carlo" supports with dependency structures (e.g. copulas).

        When calling the `run` function to do propagation, extra keyword arguments are needed to be passed down to the selected `method`.
        For example, `n_sam` for "interval_monte_carlo"; `n_slices` for "slicing"; `n_outer`, `n_inner` for "double_monte_carlo".

    example:
        >>> from pyuncertainnumber import pba
        >>> from pyuncertainnumber.propagation.p import MixedPropagation
        >>> def foo(x): return x[0] ** 3 + x[1] + x[2]
        >>> a = pba.normal([2, 3], [1])
        >>> b = pba.normal([10, 14], [1])
        >>> c = pba.normal([4, 5], [1])
        >>> mix = MixedPropagation(vars=[a,b,c], func=foo, method='slicing', interval_strategy='subinterval')
        >>> result = mix.run(n_slices=20, n_sub=2, style='endpoints')
    """

    def __init__(self, vars, func, method, dependency=None, interval_strategy=None):

        super().__init__(vars, func, method, dependency)
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

    def run(self, **kwargs):
        """doing the propagation. Extra keyword are needed to be passed down to the selected `method`"""
        match self.method:
            case "interval_monte_carlo":
                if self.dependency is None or self.dependency == "independence":
                    handler = partial(interval_monte_carlo, dependency=None)
                else:
                    imc_w_d = partial(interval_monte_carlo, dependency=self.dependency)
                    handler = imc_w_d
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

        dependency (string or Dependency): a Dependency object(i.e. a copula function) to model the dependency structure among input variables.
            Strings such as "independence" accepted for independence.

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
        dependency: str | Dependency = None,
        interval_strategy: str = None,
    ):

        self._vars = vars
        self._func = func
        self.method = method
        self.dependency = dependency
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
            self.p = AleatoryPropagation(
                self._constructs, self._func, self.method, self.dependency
            )
        elif (has_I and has_D) or has_P:
            # mixed uncertainty
            logging.info("mixed uncertainty propagation")
            self.p = MixedPropagation(
                vars=self._constructs,
                func=self._func,
                method=self.method,
                dependency=self.dependency,
                interval_strategy=self.interval_strategy,
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
        return self.p.run(**kwargs)
