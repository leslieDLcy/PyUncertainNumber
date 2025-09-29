from __future__ import annotations
from typing import TYPE_CHECKING
from scipy.stats import qmc
import numpy as np

if TYPE_CHECKING:
    from ..pba.intervals import Interval

"""some helper funcs and classes for the epistemic space"""


class EpistemicDomain:
    """Representation of the epistemic space which are indeed bounds of each dimension

    This class provides a set of handy functions to work with epistemic uncertainty in the form of bounds.
    It will be useful for tasks such as propagation or optimization where epistemic uncertainty is involved.

    args:
        vars: a set of Interval variables

    tip:
        Recommended to use for optimisation tasks where the design bounds can be quickly specified with the ``toOptBounds()`` method.

    .. seealso::

        :class:`pyuncertainnumber.src.pyuncertainnumber.opt.bo` : Bayesian optimisation class.

        :class:`pyuncertainnumber.src.pyuncertainnumber.opt.ga` : Genetic algorithm class.

    example:
        >>> from pyuncertainnumber import pba
        >>> e = EpistemicDomain(pba.I(-1, 3), pba.I(5, 9))

        >>> # convert the epistemic space to bounds for the optimizer
        >>> e.toOptBounds(method='GA')  # `varbound` for genetic algorithm
        >>> e.toOptBounds(method='BO')  # `xc_bounds` for Bayesian optimisation

        >>> # perform lhs sampling on the epistemic space
        >>> sample = e.lhs_sampling(1000)
    """

    def __init__(self, *vars: Interval):
        from ..pba.intervals.intervalOperators import make_vec_interval, parse_bounds

        try:
            self.vec_interval = make_vec_interval(vars)
        except Exception as e:
            self.vec_interval = parse_bounds(*vars)

    def lhs_sampling(self, n_samples: int):
        """perform lhs sampling on the epistemic space"""
        Xc_sampler = qmc.LatinHypercube(d=len(self.vec_interval))
        l_bounds = self.vec_interval.lo
        u_bounds = self.vec_interval.hi

        base_sample = Xc_sampler.random(n=n_samples)
        return qmc.scale(base_sample, l_bounds, u_bounds)

    def lhs_plus_endpoints(self, n_samples: int):
        """perform lhs sampling on the epistemic space and add endpoints"""
        sample = self.lhs_sampling(n_samples)
        endpoints = self.to_GA_varBounds().T
        combined_sample = np.vstack((sample, endpoints))
        return combined_sample

    def bound_rep(self):
        """return the bounds (vec or matrix) of the epistemic space"""
        return self.vec_interval

    def toOptBounds(self, method: str):
        """convert the epistemic space to bounds for the optimizer

        args:
            method (str): the optimization method to use, e.g. 'BayesOpt', 'GA'

        returns:
            the bounds of the design varibale used for the optimisation method
        """
        if method == "BO":
            return self.to_BayesOptBounds()
        elif method == "GA":
            return self.to_GA_varBounds()
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def to_GA_varBounds(self) -> np.ndarray:
        """convert the epistemic space to bounds for the genetic algorithm optimizer"""
        varbound = self.vec_interval.to_numpy()
        if varbound.ndim == 1:
            varbound = varbound.reshape(-1, 2)
        return varbound

    def to_BayesOptBounds(self, func_signature="vectorisation") -> dict:
        """convert the epistemic space to bounds for the Bayesian optimisation optimizer"""

        # new_dict = self.__dict__.copy()
        # new_dict = {k: tuple(v) for k, v in new_dict.items()}
        # return new_dict
        if func_signature == "arguments":
            return {f"x{i}": tuple(r) for i, r in enumerate(self.to_GA_varBounds())}
        elif func_signature == "vectorisation" or func_signature == "iterable":
            return self.to_GA_varBounds()
        else:
            return self.to_GA_varBounds()
