import numpy as np
import scipy.stats as sps
from pyuncertainnumber import pba
import pyuncertainnumber as pun


def test_mom():
    """test the MOM fitting method"""
    precise_sample = sps.expon(scale=1 / 0.4).rvs(15)
    imprecise_data = pba.I(lo=precise_sample - 1.4, hi=precise_sample + 1.4)

    # parametric distributional estimator using method of matching moments
    expon_fitted = pun.fit("mom", family="exponential", data=imprecise_data)
    assert isinstance(expon_fitted, pun.UncertainNumber)


def test_mle():
    """test the MLE fitting method"""
    u = pun.fit("mle", "norm", np.random.normal(0, 1, 100))
    assert isinstance(u, pun.UncertainNumber)
