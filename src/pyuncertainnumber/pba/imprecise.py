from __future__ import annotations
from typing import *
from .intervals import Interval
import scipy.stats as sps
from .ecdf import transform_eCDF_bundle, get_ecdf, eCDF_bundle


def imprecise_ecdf_sps(s: Interval) -> tuple[eCDF_bundle, eCDF_bundle]:
    """empirical cdf for interval valued data

    caveat:
        with the use of `sps.ecdf`, the probability value does not start from 0.

    returns:
        - left and right cdfs
        - pbox
    """
    b_l = transform_eCDF_bundle(sps.ecdf(s.lo))
    b_r = transform_eCDF_bundle(sps.ecdf(s.hi))

    return b_l, b_r


def imprecise_ecdf(
    s: Interval, output_type: str = "ecdf"
) -> tuple[eCDF_bundle, eCDF_bundle]:
    """Empirical cdf for interval-valued data set

    args:
        s (Interval): interval valued data set
        output_type (str): the output type, either "pbox" or "ecdf"

    returns:
        - A tuple of left and right bounding cdf

    example:
        >>> rng = np.random.default_rng(seed=42)
        >>> precise_data = rng.normal(0, 1, 10)  # precise data case
        >>> impre_data = pba.I(lo = precise_data -0.5, hi = precise_data + 0.5)
        >>> l, r = imprecise_ecdf(s=impre_data, output_type="ecdf")
    """
    from .pbox_abc import pbox_from_ecdf_bundle

    b_l = eCDF_bundle(*get_ecdf(s.lo))
    b_r = eCDF_bundle(*get_ecdf(s.hi))

    if output_type == "pbox":
        return pbox_from_ecdf_bundle(b_l, b_r)
    elif output_type == "ecdf":
        return b_l, b_r
    else:
        raise ValueError("output_type must be either 'pbox' or 'ecdf'")
