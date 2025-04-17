"""the new top-level module for the propagation of uncertain numbers"""

"""crossover logic

UncertainNumber: ops are indeed the ops for the underlying constructs

"""


from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..characterisation.uncertainNumber import UncertainNumber


from abc import ABC, abstractmethod


class P(ABC):
    def __init__(self, vars, save_results: bool = False):
        self.vars = vars
        self.save_results = save_results
        self.type_check()

    @abstractmethod
    def type_check(self):
        """if the nature of the UN suitable for the method"""
        pass


class AleatoryPropagation(P):

    from .aleatory_uncertainty.sampling_aleatory import sampling_aleatory_method

    def __init__(self):
        super().__init__()

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
    def __init__(self):
        super().__init__()

    def type_check(self):
        """only intervals"""

        from ..pba.intervals.number import Interval

        assert all(
            [isinstance(v, Interval) for v in self.vars]
        ), "Not all variables are intervals"

    def __call__(self):
        """doing the propagation"""
        pass


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

    def __call__(self):
        """doing the propagation"""
        pass
