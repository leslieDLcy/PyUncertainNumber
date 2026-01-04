import pyuncertainnumber as pun
import numpy as np
from pyuncertainnumber import pba
from pyuncertainnumber import UncertainNumber as UN
from pyuncertainnumber.pba.pbox_abc import Pbox
from pyuncertainnumber.pba.aggregation import stochastic_mixture


# *  ---------------------construction---------------------*#
def test_pun_pb_constuction():
    assert isinstance(
        pun.norm([2, 3], [0.1]), UN
    ), "Failed to construct a Pbox-type UncertainNumber object at UN level"

    assert isinstance(
        pun.I([2, 3]), UN
    ), "Failed to construct an Interval-type UncertainNumber object at UN level"


# *  ---------------------aggregation---------------------*#


def test_interval_aggregation():
    """Interval aggregation as in expert opinions"""
    lower_endpoints = np.random.uniform(-0.5, 0.5, 7)
    upper_endpoints = np.random.uniform(0.5, 1.5, 7)
    m_weights = [0.1, 0.1, 0.25, 0.15, 0.1, 0.1, 0.2]
    # a list of nInterval objects
    nI = [pba.I(*couple) for couple in zip(lower_endpoints, upper_endpoints)]

    pbox_mix = stochastic_mixture(*nI, weights=m_weights)
    print("the result of the mixture operation")
    assert isinstance(
        pbox_mix, Pbox
    ), "Failed to aggregate weights intervals expert opinions into Pbox objects"


# *  ---------------------arithmetic---------------------*#


def test_pba_arithmetic():
    """pba level"""
    a = pba.I([2, 3])  # an interval
    # _ = a.display(style="band", title="Interval [2,3]")
    b = pba.norm(0, 1)  # a precise distribution
    # _ = b.display(title="$N(0, 1)$")
    t = a + b
    assert isinstance(t, Pbox), "Failed to add a Pbox object at PBA level"


def test_UN_arithmetic():
    """UN level"""
    a = pun.norm(0, 1)
    b = pun.norm(2, 3)
    t = a + b

    assert isinstance(
        t, UN
    ), "Failed to add a Pbox-type UncertainNumber object at UN level"
