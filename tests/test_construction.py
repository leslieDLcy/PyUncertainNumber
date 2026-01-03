from pyuncertainnumber import pba
from pyuncertainnumber import UN


def test_single_parameter_construction():
    # single-parameter distribution
    a = pba.pareto(2.62)
    b_d = pba.D("pareto", 2.62)
    b = b_d.to_pbox()
    c_UN = UN(essence="distribution", distribution_parameters=["pareto", 2.62])
    c = c_UN.to_pbox()

    assert a == b and b == c and a == c, "Single-parameter construction problem"
