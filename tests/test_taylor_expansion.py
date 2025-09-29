import pytest
import jax.numpy as jnp
from pyuncertainnumber import taylor_expansion_method


def test_taylor_expansion_method():

    MEAN = jnp.array([3.0, 2.5])
    COV = jnp.array([[4, 0.3], [0.3, 0.25]])

    def bar(x):
        return x[0] ** 2 + x[1] + 3

    mu_, var_ = taylor_expansion_method(func=bar, mean=MEAN, cov=COV)
    assert mu_ == 18.5
    assert var_ == pytest.approx(179.849, rel=1e-1)
