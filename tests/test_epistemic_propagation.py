"""4 levels of epistemic propagation tests {low, b22, medium, high}"""

import pytest
import pyuncertainnumber as pun
from pyuncertainnumber import b2b, pba
from pyuncertainnumber.propagation.performance import foo_universal

""" Note

Some tests are commented out for automation but can run local tests """

# * ------------------------------ b2b level


@pytest.fixture
def example_intervals():
    a = pba.I(1, 5)
    b = pba.I(7, 13)
    c = pba.I(5, 10)
    return a, b, c


def test_b2b_level_direct_strategy(example_intervals):
    a, b, c = example_intervals

    y0 = b2b(
        vars=[a, b, c],
        func=foo_universal,
        interval_strategy="direct",
    )
    assert y0 == pba.I(13.0, 148.0)


def test_b2b_level_endpoints_strategy(example_intervals):
    a, b, c = example_intervals

    y0 = b2b(
        vars=[a, b, c],
        func=foo_universal,
        interval_strategy="endpoints",
    )
    assert y0 == pba.I(13.0, 148.0)


def test_b2b_level_subinterval_strategy(example_intervals):
    a, b, c = example_intervals

    y00 = b2b(
        vars=[a, b, c],
        func=foo_universal,
        interval_strategy="subinterval",
        subinterval_style="direct",
        n_sub=10,
    )

    y22 = b2b(
        vars=[a, b, c],
        func=foo_universal,
        interval_strategy="subinterval",
        n_sub=10,
        subinterval_style="endpoints",
    )

    assert y00 == pba.I(13.0, 148.0)
    assert y22 == pba.I(13.0, 148.0)


# def test_b2b_level_ga_opt_strategy(example_intervals):
#     a, b, c = example_intervals
#     # optimisation - ga
#     y3 = b2b(
#         vars=[a, b, c],
#         func=foo_universal,
#         interval_strategy="ga",
#     )

#     assert 12 <= y3.lo and y3.hi <= 149


# def test_b2b_level_ba_opt_strategy(example_intervals):
#     a, b, c = example_intervals

#     y3 = b2b(
#         vars=[a, b, c],
#         func=foo_universal,
#         interval_strategy="bo",
#     )

#     assert 12 <= y3.lo and y3.hi <= 149


# def test_b2b_level_cauchy_deviate_strategy(example_intervals):
#     a, b, c = example_intervals

#     y5 = b2b(
#         vars=[a, b, c],
#         func=foo_universal,
#         interval_strategy="cauchy_deviate",
#         n_sam=1000,
#     )


# * ------------------------------ high level


def test_high_level_endpoints_strategy():
    a = pun.I(1, 5)
    b = pun.I(7, 13)
    c = pun.I(5, 10)

    p = pun.Propagation(
        vars=[a, b, c],
        func=foo_universal,
        method="endpoints",
    )

    u = p.run()
    assert u.construct.lo == 13.0
    assert u.construct.hi == 148.0
