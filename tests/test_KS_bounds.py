import numpy as np
from pyuncertainnumber import pba
import pyuncertainnumber as pun
from pyuncertainnumber.pba.pbox_abc import Pbox


def test_precise_data():
    precise_data = np.random.normal(0, 1, 100)  # precise data case
    ub, lb = pba.KS_bounds(precise_data, alpha=0.025, display=True)
    p = pba.KS_bounds(precise_data, alpha=0.025, display=False, output_type="pbox")
    assert isinstance(p, Pbox)


def test_imprecise_data():
    precise_data = np.random.normal(0, 1, 100)  # precise data case
    ### imprecise data case
    impre_data = pba.I(lo=precise_data - 0.5, hi=precise_data + 0.5)
    u = pba.KS_bounds(impre_data, alpha=0.025, display=True, output_type="un")
    assert isinstance(u, pun.UncertainNumber)
