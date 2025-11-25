import numpy as np
import operator
import pytest
from pyuncertainnumber import pba


@pytest.fixture
def construct_dss():
    d1 = pba.DSS(intervals=[[1, 5], [3, 6]], masses=[0.5, 0.5])
    d2 = pba.DSS(intervals=[[2, 3], [2, 9]], masses=[0.5, 0.5])
    return d1, d2


# Map operation names to Python operator functions
OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
}


UNARY = {
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
}


@pytest.mark.parametrize("op_name", list(OPS.keys()), ids=list(OPS.keys()))
def test_arithmetic_operations(construct_dss, op_name):
    d1, d2 = construct_dss
    op = OPS[op_name]

    # keep your logic; just avoid recomputing to_pbox()
    p1 = d1.to_pbox()
    p2 = d2.to_pbox()

    c = op(p1, p2)

    # Your own style of check
    assert c == op(p1, p2)


@pytest.mark.parametrize("op_name", list(UNARY.keys()), ids=list(UNARY.keys()))
def test_unary_operations(construct_dss, op_name):
    d1, _ = construct_dss
    op = UNARY[op_name]

    p = d1.to_pbox()
    c = op(p)

    # Keep your own style of check
    assert c == op(p)
