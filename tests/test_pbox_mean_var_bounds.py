from pyuncertainnumber.pba.utils import variance_bounds_via_lp
import pytest
from pyuncertainnumber import pba
import numpy as np

pbox = pba.normal([4, 20], 1)

x_grid = np.linspace(pbox.lo, pbox.hi, 50)


def test_function_runs_without_exception():

    try:
        variance_bounds_via_lp(
            q_a=pbox.left,
            p_a=pbox.p_values,
            q_b=pbox.right,
            p_b=pbox.p_values,
            x_grid=x_grid,
            n=200,
            mu_grid=101,
        )
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")
