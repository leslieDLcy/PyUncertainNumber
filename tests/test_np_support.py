import numpy as np
import pytest

from pyuncertainnumber import pba

""" Numpy support tests"""

# *  --------------------- for pbox


@pytest.fixture
def example_staircase():
    return pba.normal(5, 1)


# unary funcs: {sin, cos, tanh, exp, log, sqrt}
def test_sin_matches_method(example_staircase):
    d = example_staircase
    assert np.sin(d) == d.sin()


def test_cos_matches_method(example_staircase):
    d = example_staircase
    assert np.cos(d) == d.cos()


def test_tanh_matches_method(example_staircase):
    d = example_staircase
    assert np.tanh(d) == d.tanh()


def test_exp_matches_method(example_staircase):
    d = example_staircase
    assert np.exp(d) == d.exp()


def test_log_matches_method(example_staircase):
    d = example_staircase
    assert np.log(d) == d.log()


def test_sqrt_matches_method(example_staircase):
    d = example_staircase
    assert np.sqrt(d) == d.sqrt()


# *  --------------------- for Interval


# *  --------------------- for Distribution


# *  --------------------- for UN
