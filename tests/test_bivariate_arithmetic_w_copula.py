import numpy as np
from pyuncertainnumber import pba
import pyuncertainnumber as pun
import operator
from pyuncertainnumber.pba.operation import copula_op

""" Test bivariate arithmetic for precise distributions with copula dependency """


def test_dist_cop():
    """Test the discretization implementation of two precise distributions under a precise copula"""

    a = pba.normal(3, 1)
    b = pba.normal(5, 1)

    # Example: Gaussian copula with correlation ρ = 0.7
    de = pba.Dependency("gaussian", 0.7)
    p_test = copula_op(a, b, de, op=operator.add)

    de = pba.Dependency("gaussian", 0.7)

    def f(XY):
        X, Y = XY
        return X + Y

    p_ref = pun.interval_monte_carlo(
        vars=[a, b],
        func=f,
        n_sam=10_000,
        dependency=de,
        interval_strategy="direct",
    )

    # specify a tolerance for the mean value
    assert p_test.sub(p_ref, "p").mean < 0.1
