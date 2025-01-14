from intervals import Interval
import scipy.stats as sps
from .utils import cdf_bundle, transform_ecdf_bundle


def imprecise_ecdf(s: Interval):
    """ empirical cdf for interval valued data

    returns:
        - left and right cdfs
        - pbox
    """
    b_l = transform_ecdf_bundle(sps.ecdf(s.lo))
    b_r = transform_ecdf_bundle(sps.ecdf(s.hi))

    return b_l, b_r
