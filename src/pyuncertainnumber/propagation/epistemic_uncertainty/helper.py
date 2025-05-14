from scipy.stats import qmc
import numpy as np


"""some helper funcs and classes for the epistemic space"""


class EpistemicDomain:
    """This function is used to represent the epistemic space which are indeed bounds of each dimension

    note:
        default setup is the Beta marginal Gaussian copula combo index from 1 meaning the first update
    """

    def __init__(self, *vars):
        from ...pba.intervals.intervalOperators import make_vec_interval

        self.vec_interval = make_vec_interval(list(vars))

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
        endpoints = self.to_OptBounds().T
        combined_sample = np.vstack((sample, endpoints))
        return combined_sample

    def bound_rep(self):
        """return the bounds of the epistemic space"""
        pass

    def to_OptBounds(self) -> np.ndarray:
        """convert the epistemic space to bounds for the optimizer"""
        return self.vec_interval.to_numpy()

    # def to_BObounds(self) -> dict:
    #     """convert the epistemic space to bounds for the Bayesian optimizer"""

    #     new_dict = self.__dict__.copy()
    #     new_dict = {k: tuple(v) for k, v in new_dict.items()}
    #     return new_dict
