from pyuncertainnumber import pba
from pyuncertainnumber import UN


def test_single_parameter_construction():
    # single-parameter distribution
    a = pba.pareto(2.62)
    b = pba.D("pareto", 2.62)
    b.to_pbox()
    c = UN(essence="distribution", distribution_parameters=["pareto", 2.62])

    assert a == b and b == c and a == c, "Single-parameter construction problem"
