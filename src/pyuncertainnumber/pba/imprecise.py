from __future__ import annotations
from typing import *
from .intervals import Interval
import scipy.stats as sps
from .utils import transform_ecdf_bundle

if TYPE_CHECKING:
    from .utils import CDF_bundle


def imprecise_ecdf(s: Interval) -> tuple[CDF_bundle, CDF_bundle]:
    """empirical cdf for interval valued data

    returns:
        - left and right cdfs
        - pbox
    """
    b_l = transform_ecdf_bundle(sps.ecdf(s.lo))
    b_r = transform_ecdf_bundle(sps.ecdf(s.hi))

    return b_l, b_r
