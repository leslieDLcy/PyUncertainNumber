from __future__ import annotations
from typing import TYPE_CHECKING
from scipy.interpolate import interp1d
import numpy as np
from .pbox_base import Pbox
from .params import Params
from ..characterisation.utils import tranform_ecdf


if TYPE_CHECKING:
    from .utils import CDF_bundle


# # TODO to integrate it
# def pbox_fromeF(a: CDF_bundle, b: CDF_bundle):
#     """pbox from emipirical CDF bundle
#     args:
#         - a : CDF bundle of lower extreme F;
#         - b : CDF bundle of upper extreme F;
#     """
#     from .pbox_abc import Staircase

#     # TODO currently the interpolation is not perfect
#     p_lo, q_lo = interpolate_p(a.probabilities, a.quantiles)
#     p_hi, q_hi = interpolate_p(b.probabilities, b.quantiles)
#     return Staircase(left=q_lo, right=q_hi)


def pbox_from_extredists(rvs, shape="beta", extre_bound_params=None):
    """transform into pbox object from extreme bounds parameterised by `sps.dist`

    args:
        rvs (list): list of scipy.stats.rv_continuous objects"""

    # x_sup
    bounds = [rv.ppf(Params.p_values) for rv in rvs]
    if extre_bound_params is not None:
        print(extre_bound_params)
    return Pbox(
        left=bounds[0],
        right=bounds[1],
        shape=shape,
    )


""" initially used for cbox next-value distribution """


def pbox_from_pseudosamples(samples):
    """a tmp constructor for pbox/cbox from approximate solution of the confidence/next value distribution

    args:
        samples (nd.array): the approximate Monte Carlo samples of the confidence/next value distribution

    note:
        ecdf is estimted from the samples and bridge to pbox/cbox
    """
    return Pbox(tranform_ecdf(samples, display=False))


def interpolate_p(p, q):
    """interpolate the cdf bundle for discrete distribution or ds structure

    note:
        - p: probabilities (x)
        - q: quantiles (y)

    return: a tuple of new (p, q)
    """

    f = interp1d(p, q, kind="next", fill_value=(p[0], p[-1]), bounds_error=False)
    # range
    new_p = np.linspace(p[0], p[-1], Params.steps)
    new_q = f(new_p)
    return new_p, new_q
