import pytest
from pyuncertainnumber import pba


# * ----------------------- mixture operation


def test_mix_dss():

    d1 = pba.DempsterShafer(intervals=[[1, 5], [3, 6]], masses=[0.5, 0.5])
    d2 = pba.DempsterShafer(intervals=[[2, 3], [2, 9]], masses=[0.5, 0.5])

    dd = pba.stochastic_mixture(d1, d2)


def test_mix_intervals():
    s = pba.stacking([[1, 3], [2, 4]], weights=[0.5, 0.5], display=True)
    new_dss = pba.DSS(intervals=[[1, 3], [2, 4]], masses=[0.5, 0.5])
    assert s == new_dss


# * ----------------------- envelope operation


def test_env_distribution():
    a = pba.normal(3, 1)
    b = pba.uniform(5, 8)
    c = pba.normal(13, 2)
    t = pba.envelope(a, b, c)
    assert a in t
    assert b in t
    assert c in t


def test_env_pbox():

    a = pba.normal([3, 5], 1)
    b = pba.uniform([3, 5], [6, 9])
    t = pba.envelope(a, b)

    assert a in t
    assert b in t


def test_env_intervals():
    a = pba.I(2, 6)
    b = pba.I(-2, 9)
    t = pba.envelope(a, b)

    assert isinstance(t, pba.Interval)
    # assert a in t
    # assert b in t


# * ----------------------- imposition operation


def test_imp_pbox():
    a = pba.normal([3, 7], 1)
    b = pba.uniform([3, 5], [6, 9])
    i = pba.imposition(a, b)
    assert i in a
    assert i in b
