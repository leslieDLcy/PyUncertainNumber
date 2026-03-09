from pyuncertainnumber import pba
from pyuncertainnumber.validation.area_metric import (
    double_metric,
    conformal_double_metric,
)


def test_double_metric():
    x_left = 7
    y_left = pba.I(14, 19)

    x_middle = pba.I(4, 9)
    y_middle = pba.I(13, 18)

    x_right = pba.I(3, 11)
    y_right = pba.I(8, 17)

    for a, b in [(x_left, y_left), (x_middle, y_middle), (x_right, y_right)]:
        print(double_metric(a, b))

    assert double_metric(x_left, y_left) == (7, 12)
    assert double_metric(x_middle, y_middle) == (9, 9)
    assert double_metric(x_right, y_right) == (5, 6)


def test_conformal_double_metric():
    x_left = 7
    y_left = pba.I(14, 19)

    x_middle = pba.I(4, 9)
    y_middle = pba.I(13, 18)

    x_right = pba.I(3, 11)
    y_right = pba.I(8, 17)

    for a, b in [(x_left, y_left), (x_middle, y_middle), (x_right, y_right)]:
        print("conformal version", conformal_double_metric(a, b))

    assert conformal_double_metric(x_left, y_left) == 12
    assert conformal_double_metric(x_middle, y_middle) == 9
    assert conformal_double_metric(x_right, y_right) == 6
