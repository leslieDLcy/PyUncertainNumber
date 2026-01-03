import pytest
import pyuncertainnumber as pun
from pyuncertainnumber import pba
from pyuncertainnumber.propagation.performance import foo_universal
import numpy as np
import pyuncertainnumber.propagation.mixed_up as mix


@pytest.fixture
def example_pboxes():
    a = pba.normal([2, 3], [1])
    b = pba.normal([10, 14], [1])
    c = pba.uniform([4, 5], [10, 11])
    return a, b, c


# * ------------------------- pba arithmetic


def test_naive_arithmetic(example_pboxes):
    a, b, c = example_pboxes

    with pba.dependency("i"):
        t = foo_universal([a, b, c])


# * ------------------------- low level


def test_low_level_imc(example_pboxes):
    """test Interval Monte Carlo method"""
    a, b, c = example_pboxes

    corre_matrix = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
    de = pba.Dependency(family="gaussian", corr=corre_matrix)
    t_gau_copula = mix.interval_monte_carlo(
        vars=[a, b, c],
        func=foo_universal,
        dependency=de,
        interval_strategy="direct",
        n_sam=10,
    )  # `direct` strategy succeeds
    assert isinstance(t_gau_copula, pun.pba.Pbox)

    t1_independence = mix.interval_monte_carlo(
        vars=[a, b, c], func=foo_universal, interval_strategy="direct", n_sam=10
    )  # `direct` strategy succeeds

    assert isinstance(t1_independence, pun.pba.Pbox)

    t1 = mix.interval_monte_carlo(
        vars=[a, b, c],
        func=foo_universal,
        interval_strategy="endpoints",
        n_sam=10,
    )  # `endpoints` strategy succeeds (expects foo_vec)
    assert isinstance(t1, pun.pba.Pbox)

    t1_sub = mix.interval_monte_carlo(
        vars=[a, b, c],
        func=foo_universal,
        interval_strategy="subinterval",
        n_sam=10,
        n_sub=10,
        subinterval_style="endpoints",
    )  # `subinterval` strategy succeeds

    assert isinstance(t1_sub, pun.pba.Pbox)


def test_low_level_slicing(example_pboxes):
    a, b, c = example_pboxes
    t3 = mix.slicing(
        vars=[a, b, c], func=foo_universal, interval_strategy="endpoints", n_slices=10
    )  # `endpoints` strategy succeeds

    assert isinstance(t3, pun.pba.Pbox)
