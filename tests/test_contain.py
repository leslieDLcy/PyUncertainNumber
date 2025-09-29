"""test if a Pbox contain another uncertain number construct"""

import pytest
from pyuncertainnumber import pba


@pytest.fixture
def normal_pba():
    return pba.normal([3, 5], 1)


def test_contain_realnumber(normal_pba):
    n1 = 7
    n2 = 20
    assert n1 in normal_pba
    assert n2 not in normal_pba


def test_contain_interval(normal_pba):
    i1 = pba.I(2, 4)
    i2 = pba.I(-1, 5)
    assert i1 in normal_pba
    assert i2 not in normal_pba


def test_contain_distribution(normal_pba):
    d1 = pba.normal(4, 1)
    d2 = pba.normal(2, 1)
    assert d1 in normal_pba
    assert d2 not in normal_pba
