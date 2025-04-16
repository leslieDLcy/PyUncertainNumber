from __future__ import annotations
from .intervals import Interval as I
from .pbox_base import Pbox
from .utils import NotIncreasingError
from typing import *
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
from .params import Params
from .logical import sometimes
from .utils import pl_ecdf_bounding_bundles, weighted_ecdf, CDF_bundle
from .imprecise import imprecise_ecdf
from .pbox_abc import Staircase
from numbers import Number

""" non-parametric pbox  """


__all__ = [
    "known_constraints",
    "min_max",
    "min_max_mean",
    "min_mean",
    "min_max_mean_std",
    "min_max_mean_var",
    "min_max_mode",
    "min_max_median",
    # "min_max_median_is_mode",
    "mean_std",
    "mean_var",
    "pos_mean_std",
    # "symmetric_mean_std",
    "from_percentiles",
    "KS_bounds",
]
# ---------------------from data---------------------#

if TYPE_CHECKING:
    from .utils import CDF_bundle
    from .pbox_base import Pbox


def KS_bounds(s, alpha: float, display=True) -> CDF_bundle:
    """construct free pbox from sample data by Kolmogorov-Smirnoff confidence bounds

    args:
        - s (array-like): sample data, precise and imprecise
        - dn (scalar): KS critical value at a significance level and sample size N;
    """
    # TODO quantile of two bounds have different support ergo not a box yet
    # * to make the output as a pbox
    dn = d_alpha(len(s), alpha)
    # precise data
    if isinstance(s, list | np.ndarray):
        # ecdf = sps.ecdf(s)
        # b = transform_ecdf_bundle(ecdf)

        q, p = weighted_ecdf(s)
        f_l, f_r = p + dn, p - dn
        f_l, f_r = logical_bounding(f_l), logical_bounding(f_r)
        # new ecdf bundles
        b_l, b_r = CDF_bundle(q, f_l), CDF_bundle(q, f_r)

        if display:
            fig, ax = plt.subplots()
            ax.step(q, p, color="black", ls=":", where="post")
            pl_ecdf_bounding_bundles(b_l, b_r, ax=ax)
        return b_l, b_r
    # imprecise data
    elif isinstance(s, I):
        b_l, b_r = imprecise_ecdf(s)
        b_lbp, b_rbp = imprecise_ecdf(s)

        b_l.probabilities += dn
        b_r.probabilities -= dn

        b_l.probabilities, b_r.probabilities = logical_bounding(
            b_l.probabilities
        ), logical_bounding(b_r.probabilities)

        if display:
            fig, ax = plt.subplots()
            # plot the epimirical ecdf
            ax.plot(
                b_lbp.quantiles,
                b_lbp.probabilities,
                drawstyle="steps-post",
                ls=":",
                color="gray",
            )
            ax.plot(
                b_rbp.quantiles,
                b_rbp.probabilities,
                drawstyle="steps-post",
                ls=":",
                color="gray",
            )

            # plot the KS bounds
            pl_ecdf_bounding_bundles(
                b_l,
                b_r,
                alpha,
                ax,
                title=f"Kolmogorov-Smirnoff confidence bounds at {(1-2*alpha)*100}% confidence level",
            )
    else:
        raise ValueError("Invalid input data type")
    return b_l, b_r


def logical_bounding(a):
    """Sudret p16. eq(2.21)"""
    a = np.where(a < 0, 0, a)
    a = np.where(a < 1, a, 1)
    return a


def d_alpha(n, alpha):
    """compute the Smirnov critical value for a given sample size and significance level

    note:
        Tretiak p12. eq(8): alpha = (1-c) / 2 where c is the confidence level

    args:
        - n (int): sample size;
        - alpha (float): significance level;
    """

    A = {0.1: 0.00256, 0.05: 0.05256, 0.025: 0.11282}
    return (
        np.sqrt(np.log(1 / alpha) / (2 * n))
        - 0.16693 * (1 / n)
        - A.get(alpha, 1000) * (n ** (-3 / 2))
    )


# * ---------top level func for known statistical properties------*#


def known_constraints(
    maximum=None,
    mean=None,
    median=None,
    minimum=None,
    mode=None,
    percentiles=None,
    std=None,
    var=None,
) -> Pbox:
    args = {
        "maximum": maximum,
        "mean": mean,
        "median": median,
        "minimum": minimum,
        "mode": mode,
        "percentiles": percentiles,
        "std": std,
        "var": var,
    }
    shape_control = ["percentiles", "symmetric"]
    present_keys = tuple(
        sorted(k for k, v in args.items() if v is not None if k not in shape_control)
    )

    # template:
    # ('a', 'b'): handle_ab,

    routes = {
        ("percentiles"): from_percentiles,
        ("maximum", "minimum"): min_max,
        ("mean", "minimum"): min_mean,
        ("maximum", "mean"): max_mean,
        ("mean", "std"): mean_std,
        ("mean", "var"): mean_var,
        ("maximum", "mean", "minimum"): min_max_mean,
        ("maximum", "minimum", "mode"): min_max_mode,
        ("maximum", "median", "minimum"): min_max_median,
        ("maximum", "mean", "minimum", "std"): min_max_mean_std,
        ("maximum", "mean", "minimum", "var"): min_max_mean_var,
    }

    handler1 = routes.get(present_keys, handle_default)
    base_pbox = handler1(**{k: args[k] for k in present_keys})

    # second-level shape control to see if percentiles or some other constraints are present
    further_shape_controls = [
        k for k, v in args.items() if v is not None if k in shape_control
    ]

    if not further_shape_controls:
        return base_pbox
    else:
        for c_keys in further_shape_controls:
            c_handler = routes.get(c_keys, handle_default)
            c_pbox = c_handler(args[c_keys])
            if not isinstance(base_pbox, Pbox):
                return c_pbox
            imp_pbox = base_pbox.imp(c_pbox)
        return imp_pbox


def handle_default(**kwargs):
    return f"No match. Received: {kwargs}"


# * --------------------- supporting functions---------------------*#


def min_max(minimum: Number, maximum: Number) -> Staircase:
    """Equivalent to an interval object constructed as a nonparametric Pbox.

    args:
        minimum : Left end of box
        maximum : Right end of box

    returns: Pbox
    """

    return Staircase(
        left=np.repeat(minimum, Params.steps),
        right=np.repeat(maximum, Params.steps),
        mean=I(minimum, maximum),
        var=I(0, (maximum - minimum) * (maximum - minimum) / 4),
    )


def min_mean(minimum, mean, steps=Params.steps) -> Staircase:
    """Nonparametric pbox construction based on constraint of minimum and mean

    args:
        minimum (number): minimum value of the variable
        mean (number): mean value of the variable

    return:
        Pbox
    """
    jjj = np.array([j / steps for j in range(1, steps - 1)] + [1 - 1 / steps])
    right = [((mean - minimum) / (1 - j) + minimum) for j in jjj]

    return Staircase(
        left=np.repeat(minimum, Params.steps),
        right=right,
        mean=I(mean, mean),
    )


def max_mean(
    maximum: Number,
    mean: Number,
    steps=Params.steps,
) -> Staircase:
    # TODO no __neg__
    """Nonparametric pbox construction based on constraint of maximum and mean

    args:
        maximum (number): maximum value of the variable
        mean (number): mean value of the variable

    return:
        Pbox
    """
    return min_mean(-maximum, -mean).__neg__()


def mean_std(mean: Number, std: Number, steps=Params.steps) -> Staircase:
    """Nonparametric pbox construction based on constraint of mean and std

    args:
        mean (number): mean value of the variable
        std (number): std value of the variable

    return:
        Pbox
    """
    iii = [1 / steps] + [i / steps for i in range(1, steps - 1)]
    jjj = [j / steps for j in range(1, steps - 1)] + [1 - 1 / steps]

    left = [mean - std * np.sqrt(1 / i - 1) for i in iii]
    right = [mean + std * np.sqrt(j / (1 - j)) for j in jjj]
    return Staircase(left=left, right=right, mean=I(mean, mean), var=I(std**2, std**2))


def mean_var(
    mean: Number,
    var: Number,
) -> Staircase:
    """Nonparametric pbox construction based on constraint of mean and var

    args:
        mean (number): mean value of the variable
        vasr (number): var value of the variable

    return:
        Pbox
    """
    return mean_std(mean, np.sqrt(var))


def min_max_mean(
    minimum: Number,
    maximum: Number,
    mean: Number,
    steps: int = Params.steps,
) -> Staircase:
    # TODO var is missing
    """
    Generates a distribution-free p-box based upon the minimum, maximum and mean of the variable

    **Parameters**:

        ``minimum`` : minimum value of the variable

        ``maximum`` : maximum value of the variable

        ``mean`` : mean value of the variable


    **Returns**:

        ``Pbox``
    """
    mid = (maximum - mean) / (maximum - minimum)
    ii = [i / steps for i in range(steps)]
    left = [minimum if i <= mid else ((mean - maximum) / i + maximum) for i in ii]
    jj = [j / steps for j in range(1, steps + 1)]
    right = [maximum if mid <= j else (mean - minimum * j) / (1 - j) for j in jj]
    # print(len(left))
    return Staircase(
        left=np.array(left), right=np.array(right), mean=I(mean, mean), steps=steps
    )


def pos_mean_std(
    mean: Union[nInterval, float, int],
    std: Union[nInterval, float, int],
    steps=Params.steps,
) -> Pbox:
    """
    Generates a positive distribution-free p-box based upon the mean and standard deviation of the variable

    **Parameters**:

        ``mean`` : mean of the variable

        ``std`` : standard deviation of the variable


    **Returns**:

        ``Pbox``

    """
    iii = [1 / steps] + [i / steps for i in range(1, steps - 1)]
    jjj = [j / steps for j in range(1, steps - 1)] + [1 - 1 / steps]

    left = [max((0, mean - std * np.sqrt(1 / i - 1))) for i in iii]
    right = [min((mean / (1 - j), mean + std * np.sqrt(j / (1 - j)))) for j in jjj]

    return Staircase(
        left=left,
        right=right,
        steps=steps,
        mean=I(mean, mean),
        var=I(std**2, std**2),
    )


def min_max_mode(
    minimum: Number,
    maximum: Number,
    mode: Number,
    steps: int = Params.steps,
) -> Staircase:
    """Nonparametric pbox construction based on constraint of mean and var

    args:
        minimum: minimum value of the variable
        maximum: maximum value of the variable
        mode (number): mode value of the variable

    return:
        Pbox
    """
    if minimum == maximum:
        return min_max(minimum, maximum)

    ii = np.array([i / steps for i in range(steps)])
    jj = np.array([j / steps for j in range(1, steps + 1)])

    l = ii * (mode - minimum) + minimum
    r = jj * (maximum - mode) + mode
    mean_l = (minimum + mode) / 2
    mean_r = (mode + maximum) / 2
    var_l = 0
    var_r = (maximum - minimum) * (maximum - minimum) / 12

    return Staircase(left=l, right=r, mean=I(mean_l, mean_r), var=I(var_l, var_r))


def min_max_median(
    minimum: Union[nInterval, float, int],
    maximum: Union[nInterval, float, int],
    median: Union[nInterval, float, int],
    steps: int = Params.steps,
) -> Pbox:
    # TODO error in function
    """
    Generates a distribution-free p-box based upon the minimum, maximum and median of the variable

    **Parameters**:

        ``minimum`` : minimum value of the variable

        ``maximum`` : maximum value of the variable

        ``median`` : median value of the variable


    **Returns**:

        ``Pbox``

    """
    if minimum == maximum:
        return min_max(minimum, maximum)

    ii = np.array([i / steps for i in range(steps)])
    jj = np.array([j / steps for j in range(1, steps + 1)])

    return Staircase(
        left=np.array([p if p > 0.5 else minimum for p in ii]),
        right=np.array([p if p <= 0.5 else minimum for p in jj]),
        mean=I((minimum + median) / 2, (median + maximum) / 2),
        var=I(0, (maximum - minimum) * (maximum - minimum) / 4),
    )


# TODO not updated yet
# def min_max_median_is_mode(
#     minimum: Union[nInterval, float, int],
#     maximum: Union[nInterval, float, int],
#     m: Union[nInterval, float, int],
#     steps: int = Params.steps,
# ) -> Pbox:
#
#     """
#     Generates a distribution-free p-box based upon the minimum, maximum and median/mode of the variable when median = mode.

#     **Parameters**:

#         ``minimum`` : minimum value of the variable

#         ``maximum`` : maximum value of the variable

#         ``m`` : m = median = mode value of the variable


#     **Returns**:

#         ``Pbox``

#     """
#     ii = np.array([i / steps for i in range(steps)])
#     jjj = [j / steps for j in range(1, steps - 1)] + [1 - 1 / steps]

#     u = [p * 2 * (m - minimum) + minimum if p <= 0.5 else m for p in ii]

#     d = [(p - 0.5) * 2 * (maximum - m) + m if p > 0.5 else m for p in jjj]

#     return Pbox(
#         left=u,
#         right=d,
#         mean_left=(minimum + 3 + m) / 4,
#         mean_right=(3 * m + maximum) / 4,
#         var_left=0,
#         var_right=(maximum - minimum) * (maximum - minimum) / 4,
#     )

# TODO not updated yet
# def symmetric_mean_std(
#     mean: Union[nInterval, float, int],
#     std: Union[nInterval, float, int],
#     steps: int = Params.steps,
# ) -> Pbox:
#     """
#     Generates a symmetrix distribution-free p-box based upon the mean and standard deviation of the variable

#     **Parameters**:

#     ``mean`` :  mean value of the variable
#     ``std`` : standard deviation of the variable

#     **Returns**

#         ``Pbox``

#     """
#     iii = [1 / steps] + [i / steps for i in range(1, steps - 1)]
#     jjj = [j / steps for j in range(1, steps - 1)] + [1 - 1 / steps]

#     u = [mean - std / np.sqrt(2 * p) if p <= 0.5 else mean for p in iii]
#     d = [mean + std / np.sqrt(2 * (1 - p)) if p > 0.5 else mean for p in jjj]

#     return Pbox(
#         left=u,
#         right=d,
#         mean_left=mean,
#         mean_right=mean,
#         var_left=std**2,
#         var_right=std**2,
#     )


def min_max_mean_std(
    minimum: Number,
    maximum: Number,
    mean: Number,
    std: Number,
    **kwargs,
) -> Staircase:
    """
    Generates a distribution-free p-box based upon the minimum, maximum, mean and standard deviation of the variable

    **Parameters**

        ``minimum`` : minimum value of the variable
        ``maximum`` : maximum value of the variable
        ``mean`` : mean value of the variable
        ``std`` :standard deviation of the variable

    **Returns**

        ``Pbox``

    .. seealso::

        :func:`min_max_mean_var`

    """
    if minimum == maximum:
        return min_max(minimum, maximum)

    def _left(x):

        if isinstance(x, (int, float, np.number)):
            return x
        if x.__class__.__name__ == "Interval":
            return x.left
        if x.__class__.__name__ == "Pbox":
            return min(x.left)
        else:
            raise Exception("wrong type encountered")

    def _right(x):
        if isinstance(x, (int, float, np.number)):
            return x
        if x.__class__.__name__ == "Interval":
            return x.right
        if x.__class__.__name__ == "Pbox":
            return max(x.right)

    def _imp(a, b):
        return nInterval(max(_left(a), _left(b)), min(_right(a), _right(b)))

    def _env(a, b):
        return nInterval(min(_left(a), _left(b)), max(_right(a), _right(b)))

    def _constrain(a, b, msg):
        if (_right(a) < _left(b)) or (_right(b) < _left(a)):
            print("Math Problem: impossible constraint", msg)
        return _imp(a, b)

    zero = 0.0
    one = 1.0
    ran = maximum - minimum
    m = _constrain(mean, nInterval(minimum, maximum), "(mean)")
    s = _constrain(
        std,
        _env(
            nInterval(0.0),
            (abs(ran * ran / 4.0 - (maximum - mean - ran / 2.0) ** 2)) ** 0.5,
        ),
        " (dispersion)",
    )
    ml = (m.left - minimum) / ran
    sl = s.left / ran
    mr = (m.right - minimum) / ran
    sr = s.right / ran
    z = min_max(minimum, maximum)
    n = len(z.left)
    L = [0.0] * n
    R = [1.0] * n
    for i in range(n):
        p = i / n
        if p <= zero:
            x2 = zero
        else:
            x2 = ml - sr * (one / p - one) ** 0.5
        if ml + p <= one:
            x3 = zero
        else:
            x5 = p * p + sl * sl - p
            if x5 >= zero:
                x4 = one - p + x5**0.5
                if x4 < ml:
                    x4 = ml
            else:
                x4 = ml
            x3 = (p + sl * sl + x4 * x4 - one) / (x4 + p - one)
        if (p <= zero) or (p <= (one - ml)):
            x6 = zero
        else:
            x6 = (ml - one) / p + one
        L[i] = max(max(max(x2, x3), x6), zero) * ran + minimum

        p = (i + 1) / n
        if p >= one:
            x2 = one
        else:
            x2 = mr + sr * (one / (one / p - one)) ** 0.5
        if mr + p >= one:
            x3 = one
        else:
            x5 = p * p + sl * sl - p
            if x5 >= zero:
                x4 = one - p - x5**0.5
                if x4 > mr:
                    x4 = mr
            else:
                x4 = mr
            x3 = (p + sl * sl + x4 * x4 - one) / (x4 + p - one) - one

        if ((one - mr) <= p) or (one <= p):
            x6 = one
        else:
            x6 = mr / (one - p)
        R[i] = min(min(min(x2, x3), x6), one) * ran + minimum

    v = s**2
    return Staircase(
        left=np.array(L),
        right=np.array(R),
        mean=I(_left(m), _right(m)),
        var=I(_left(v), _right(v)),
        **kwargs,
    )


def min_max_mean_var(
    minimum: Number,
    maximum: Number,
    mean: Number,
    var: Number,
    **kwargs,
) -> Staircase:
    """
    Generates a distribution-free p-box based upon the minimum, maximum, mean and standard deviation of the variable

    **Parameters**

        ``minimum`` : minimum value of the variable
        ``maximum`` : maximum value of the variable
        ``mean`` : mean value of the variable
        ``var`` :variance of the variable

    **Returns**

        ``Pbox``


    .. admonition:: Implementation

        Equivalent to ``min_max_mean_std(minimum,maximum,mean,np.sqrt(var))``

    .. seealso::

        :func:`min_max_mean_std`

    """
    return min_max_mean_std(minimum, maximum, mean, np.sqrt(var), **kwargs)


def from_percentiles(percentiles: dict, steps: int = Params.steps) -> Pbox:
    """yields a distribution-free p-box based on specified percentiles of the variable

    args:
        ``percentiles`` : dictionary of percentiles and their values (e.g. {0: 0, 0.1: 1, 0.5: 2, 0.9: nInterval(3,4), 1:5})
        ``steps`` : number of steps to use in the p-box

    .. important::

        The percentiles dictionary is of the form {percentile: value}. Where value can either be a number or an nInterval. If value is a number, the percentile is assumed to be a point percentile. If value is an nInterval, the percentile is assumed to be an interval percentile.

    .. warning::

        If no keys for 0 and 1 are given, ``-np.inf`` and ``np.inf`` are used respectively. This will result in a p-box that is not bounded and raise a warning.

        If the percentiles are not increasing, the percentiles will be intersected. This may not be desired behaviour.

    .. error::

        ``ValueError``: If any of the percentiles are not between 0 and 1.

    **Returns**

        ``Pbox``


    **Example**:

    .. code-block:: python

        pba.from_percentiles(
            {0: 0,
            0.25: 0.5,
            0.5: pba.I(1,2),
            0.75: pba.I(1.5,2.5),
            1: 3}
        ).show()
    """
    # check if 0 and 1 are in the dictionary
    if 0 not in percentiles.keys():
        percentiles[0] = -np.inf
        warn("No value given for 0 percentile. Using -np.inf")
    if 1 not in percentiles.keys():
        percentiles[1] = np.inf
        warn("No value given for 1 percentile. Using np.inf")

    # sort the dictionary by percentile
    percentiles = dict(sorted(percentiles.items()))

    # transform values to intervals
    for k, v in percentiles.items():
        if not isinstance(v, I):
            percentiles[k] = I(v, v)

    if any([p < 0 or p > 1 for p in percentiles.keys()]):
        raise ValueError("Percentiles must be between 0 and 1")

    left = []
    right = []
    for i in np.linspace(0, 1, steps):
        smallest_key = min(key for key in percentiles.keys() if key >= i)
        largest_key = max(key for key in percentiles.keys() if key <= i)
        left.append(percentiles[largest_key].left)
        right.append(percentiles[smallest_key].right)

    try:
        # return Pbox(left, right, steps=steps, interpolation="outer")  # backup
        return Staircase(left=left, right=right, steps=steps)
    except NotIncreasingError:
        warn("Percentiles are not increasing. Will take intersection of percentiles.")

        left = []
        right = []
        p = list(percentiles.keys())
        for i, j, k in zip(p, p[1:], p[2:]):
            if sometimes(percentiles[j] < percentiles[i]):
                percentiles[j] = nInterval(percentiles[i].right, percentiles[j].right)
            if sometimes(percentiles[j] > percentiles[k]):
                percentiles[j] = nInterval(percentiles[j].left, percentiles[k].left)

        left = []
        right = []
        for i in np.linspace(0, 1, steps):
            smallest_key = min(key for key in percentiles.keys() if key >= i)
            left.append(percentiles[smallest_key].left)
            right.append(percentiles[smallest_key].right)

        # return Pbox(left, right, steps=steps, interpolation="outer")  # backup
        return Staircase(left=left, right=right, steps=steps)
    except:
        raise Exception("Unable to generate p-box")
