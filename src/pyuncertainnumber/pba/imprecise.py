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


def imprecise_ecdf(s: Interval) -> tuple[eCDF_bundle, eCDF_bundle]:
    """empirical cdf for interval valued data

    returns:
        - left and right cdfs
        - pbox
    """
    b_l = eCDF_bundle(*get_ecdf(s.lo))
    b_r = eCDF_bundle(*get_ecdf(s.hi))

    return b_l, b_r
