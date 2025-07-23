from pyuncertainnumber import pba


def test_single_parameter_construction():
    # single-parameter distribution
    a = pba.pareto(2.62)

    b = pba.D("pareto", 2.62)
    b.to_pbox()

    assert a == b, "Single-parameter construction problem"
