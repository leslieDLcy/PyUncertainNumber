"""a Cbox constructor by Leslie"""

import numpy as np
from .params import Params
from ..characterisation.utils import tranform_ecdf


class Cbox:
    """
    Confidence boxes (c-boxes) are imprecise generalisations of traditional confidence distributions

    They have a different interpretation to p-boxes but rely on the same underlying mathematics.
    As such in pba-for-python c-boxes inhert most of their methods from Pbox.

    Args:
        Pbox (_type_): _description_
    """

    def __init__(self, *args, extre_bound_params=None, **kwargs):
        """Cbox constructor

        args:
            extre_bound_params: envelope (extreme) bounds of the box
        """
        self.extre_bound_params = extre_bound_params
        super().__init__(*args, **kwargs)

    def __repr__(self):
        # notation is defined as two bounding c.d.fs

        if self.extre_bound_params is None:
            return f"Cbox ~ approximation"
        if len(self.extre_bound_params) > 1:
            return f"Cbox ~ [{self.shape}{self.extre_bound_params[0]}, {self.shape}{self.extre_bound_params[1]}]"
        else:
            return f"Cbox ~ {self.shape}{self.extre_bound_params[0]}"

    def display(self, parameter_name=None, **kwargs):
        if parameter_name is not None:
            ax = super().display(
                title=f"Cbox {parameter_name}", fill_color="salmon", **kwargs
            )
        else:
            ax = super().display(fill_color="salmon", **kwargs)
        ax.set_ylabel("Confidence")
        return ax

    def ci(self, c=0.95, alpha=None, beta=None, style="two-sided"):
        """query the confidence interval at a given confidence level `c`"""
        if style == "two-sided":
            if alpha is None:
                alpha = (1 - c) / 2
            if beta is None:
                beta = 1 - (1 - c) / 2

            l_quantile = self.cuth(alpha).left
            r_quantile = self.cuth(beta).right
        if style == "one-sides":
            l_quantile = self.cuth(0.01).left
            r_quantile = self.cuth(c).right

        return np.array([l_quantile, r_quantile])

    # def query_confidence(self, level=None, low=None, upper=None):

    #     """ or simply the `ci` function

    #     note:
    #         to return the symmetric confidence interval
    #     """
    #     if level is not None:
    #         low = (1-level)/2
    #         upper = 1-low

    #     return self.left(low), self.right(upper)


# * ---------------------  constructors--------------------- *#


def cbox_from_extredists(rvs, shape=None, extre_bound_params=None):
    """define cbox via parameterised extreme bouding distrbution functions

    args:
        rvs (list): list of `scipy.stats.rv_continuous` objects
        extre_bound_params (list): list of parameters for the extreme bounding c.d.f
    """
    if not isinstance(rvs, list | tuple):
        rvs = [rvs]
    # extreme bouding quantiles
    bounds = [rv.ppf(Params.p_values) for rv in rvs]
    # if extre_bound_params is not None: print(extre_bound_params)
    if not isinstance(extre_bound_params, list):
        extre_bound_params = [extre_bound_params]

    return Cbox(
        *bounds,
        extre_bound_params=extre_bound_params,
        shape=shape,
    )


# used for nextvalue distribution which by nature is pbox


def cbox_from_pseudosamples(samples):

    return Cbox(tranform_ecdf(samples, display=False))
