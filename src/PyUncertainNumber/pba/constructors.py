
from scipy.interpolate import interp1d
import numpy as np


def pbox_fromDiscreteF(a, b):
    """ pbox from discrete CDF bundle 
    args:
        - a : CDF bundle of lower extreme F;
        - b : CDF bundle of upper extreme F;
    """
    from .pbox_base import Pbox
    p_lo, q_lo = interpolate_p(a.probability, a.quantile)
    p_hi, q_hi = interpolate_p(b.probability, b.quantile)
    return Pbox(left=q_lo, right=q_hi)


def interpolate_p(x, y):
    """ interpolate the cdf bundle for discrete distribution or ds structure 
    note:
        - x: probabilities
        - y: quantiles
    """

    f = interp1d(x, y,  kind='next', fill_value=(
        x[0], x[-1]), bounds_error=False)
    # range
    q = np.linspace(x[0], x[-1], 200)
    p = f(q)
    return q, p
