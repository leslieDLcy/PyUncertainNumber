import numpy as np
from numpy.typing import ArrayLike
from numbers import Number
from pyuncertainnumber import Pbox, Interval
from pyuncertainnumber.pba.core import area_metric_ecdf
import scipy.stats as sps
import matplotlib.pyplot as plt


def am_distance_counter(A, B):
    """Compute the distance between two sets of intervals as used in the area metric.

    notes:
        It is essentially doing:

        def f(a, b, c, d):
            return np.maximum.reduce([c - b, a - d, 0])

    """
    a, b = A[:, 0], A[:, 1]
    c, d = B[:, 0], B[:, 1]
    return np.maximum.reduce([c - b, a - d, np.zeros_like(a)])


def function_succeeds(f, *args, **kwargs):
    try:
        f(*args, **kwargs)
        return True
    except Exception:
        return False


def area_metric_pbox(a: Pbox, b: Pbox):
    """when a and b are both Pboxes"""
    # diff = endpoint_distance(a.to_numpy(), b.to_numpy())  # old version busted by Scott
    diff = am_distance_counter(a.to_numpy(), b.to_numpy())
    return np.trapz(y=diff, x=a.p_values)


def area_metric_pbox_diff(a: Pbox, b: Pbox):
    """when a and b are both Pboxes"""
    # diff = endpoint_distance(a.to_numpy(), b.to_numpy())  # old version busted by Scott
    diff = am_distance_counter(a.to_numpy(), b.to_numpy())
    return diff


def area_metric_sample(a: ArrayLike, b: ArrayLike):
    return sps.wasserstein_distance(a, b)


#! not in use.
def area_metric_np_numbers(a, b):
    """when a and b are both numpy arrays of scalar numbers, compute the area metric accordingly"""
    assert np.isscalar(a) and np.isscalar(b), "Both a and b must be scalar numbers."
    return abs(a - b)


def area_metric_number(a: Pbox | Number, b: Pbox | Number) -> float:
    """if any of a or b is a number, compute area metric accordingly"""
    from pyuncertainnumber import pba

    if isinstance(a, Number) and isinstance(b, Number):
        return abs(a - b)
    if isinstance(a, Number):
        a, b = b, a  # swap so b is the number
    if isinstance(a, Pbox) and a.degenerate:
        return area_metric_ecdf(a.left, b, a.p_values)
    if isinstance(a, Pbox) and (not a.degenerate):
        # make b a Pbox
        b = pba.I(b).to_pbox()
        return area_metric_pbox(a, b)


def area_metric(a: Number | Pbox | ArrayLike, b: Number | Pbox | ArrayLike) -> float:
    """Compute the area metric between two objects.

    note:
        top-level function to compute area metric between any two objects
    """
    if isinstance(a, Number) or isinstance(b, Number):
        return area_metric_number(a, b)
    if isinstance(a, Pbox) and isinstance(b, Pbox):
        if a.degenerate and b.degenerate:
            return area_metric_ecdf(a.left, b.left, a.p_values)
        elif function_succeeds(a.imp, b):
            return 0.0
        else:
            return area_metric_pbox(a, b)
    if isinstance(a, (np.ndarray, list)) and isinstance(b, (np.ndarray, list)):
        return area_metric_sample(a, b)
    elif (isinstance(a, Pbox) and isinstance(b, np.ndarray)) or (
        isinstance(a, np.ndarray) and isinstance(b, Pbox)
    ):
        # make a a Pbox and b a sample anyway
        if not isinstance(a, Pbox):
            a, b = b, a

        # b has to be a scalar sample
        b = np.squeeze(b).item()
        return area_metric_number(a, b)
    else:
        raise NotImplementedError("Area metric not implemented for these types.")


def double_metric(p: Number | Pbox | Interval, o: Number | Pbox | Interval):
    """Double metric for two uncertain numbers.

    args:
        p: a prediction uncertain number (Pbox)
        o: an observation uncertain number (Pbox or scalar)

    note:
        Typical case is for validation between prediction and observation where both are uncertain numbers.

    """
    if isinstance(p, Number):
        p = Interval(p)

    if isinstance(o, Number):
        o = Interval(o)

    return area_metric(p.left, o.left), area_metric(p.right, o.right)


def conformal_double_metric(p: Number | Pbox | Interval, o: Number | Pbox | Interval):
    """Propsed conformal version of the double metric, which takes the maximum of the two area metrics."""
    return max(double_metric(p, o))


# * --------------------------- area metric hints plot


def am_diff_register(A, B, debug=False):
    """Register the distance and return a structured array with fields:

    returns:
        In each tuple of the output array:
        - distance: float (max of c-b, a-d, 0)
        - x1: first endpoint (c or a, or 0 if 0 wins)
        - x2: second endpoint (b or d, or 0 if 0 wins)
    """
    A = np.asarray(A)
    B = np.asarray(B)

    # Expect shape (N, 2): [:,0]=start=a/c, [:,1]=end=b/d
    a, b = A[:, 0], A[:, 1]
    c, d = B[:, 0], B[:, 1]

    c_minus_b = c - b
    a_minus_d = a - d
    zeros = np.zeros_like(a, dtype=np.result_type(a, b, c, d, float))

    # Stack and argmax exactly mirrors "max(c-b, a-d, 0)" with deterministic tie-breaking.
    stacked = np.stack([c_minus_b, a_minus_d, zeros], axis=1)
    max_indices = np.argmax(stacked, axis=1)
    distances = stacked[np.arange(len(a)), max_indices]

    # Structured result
    result = np.zeros(len(a), dtype=[("distance", "f8"), ("x1", "f8"), ("x2", "f8")])
    result["distance"] = distances

    mask_cb = max_indices == 0
    mask_ad = max_indices == 1
    mask_0 = max_indices == 2

    result["x1"][mask_cb] = c[mask_cb]
    result["x2"][mask_cb] = b[mask_cb]

    result["x1"][mask_ad] = a[mask_ad]
    result["x2"][mask_ad] = d[mask_ad]

    result["x1"][mask_0] = 0.0
    result["x2"][mask_0] = 0.0

    if debug:
        print("a:", a)
        print("b:", b)
        print("c:", c)
        print("d:", d)
        print("c-b:", c_minus_b)
        print("a-d:", a_minus_d)
        print("stacked:\n", stacked)
        print("argmax:", max_indices)
        print("distances:", distances)
        print("result:", result)

    return result


def intervals_from_res(res):
    """
    From a structured array `res` with fields: 'distance', 'x1', 'x2',
    return:
      - mask: boolean mask where distance != 0
      - intervals: (N, 2) array of sorted [lo, hi] endpoints for rows where mask is True
    """
    mask = res["distance"] != 0
    x1 = res["x1"][mask]
    x2 = res["x2"][mask]
    lo = np.minimum(x1, x2)
    hi = np.maximum(x1, x2)
    intervals = np.stack([lo, hi], axis=1)
    return mask, intervals


def plot_intervals_from_res(
    res, p, *, show_points=False, title=None, include_ties=False, ax=None
):
    """Plot horizontal intervals from `res` at heights given by `p`.

    args
        res (ArrayLike) : Structured array with fields ('distance', 'x1', 'x2').

        p (ArrayLike) : 1D array of same length as `res`, giving y-coordinate (height) of each interval.

        show_points (Boolean): If True, shows markers at interval endpoints.

        title (str): Plot title. Optional

        include_ties (bool): If True, also include intervals where distance == 0 but argmax picked c-b or a-d.

        ax (matplotlib.axes.Axes) :  Axis to plot on. If None, a new figure and axis are created. Optional

    returns
        ax (matplotlib.axes.Axes): The matplotlib axis used for plotting.
    """
    res = np.asarray(res)
    p = np.asarray(p)
    if res.shape[0] != p.shape[0]:
        raise ValueError("p must have the same number of rows as res")

    # Select rows to plot
    if include_ties:
        mask = ~((res["x1"] == 0) & (res["x2"] == 0))
    else:
        mask = res["distance"] > 0

    if ax is None:
        fig, ax = plt.subplots()

    if not np.any(mask):
        ax.set_xlabel("x")
        ax.set_ylabel("p (height)")
        if title:
            ax.set_title(title)
        return ax

    x1 = res["x1"][mask]
    x2 = res["x2"][mask]
    y = p[mask]

    lo = np.minimum(x1, x2)
    hi = np.maximum(x1, x2)

    # Draw intervals
    for xi0, xi1, yi in zip(lo, hi, y):
        ax.plot([xi0, xi1], [yi, yi], color="lavender")  # horizontal segment
        if show_points:
            ax.plot([xi0, xi1], [yi, yi], "o", color="C0")

    ax.set_xlabel("x")
    ax.set_ylabel("p (height)")
    if title:
        ax.set_title(title)

    # Set limits dynamically based on data
    ax.set_xlim(lo.min() - 0.1, hi.max() + 0.1)
    ax.set_ylim(y.min() - 0.1, y.max() + 0.1)

    return ax


def integrate_distance(res, p):
    """
    Integrate distance vs p with the trapezoidal rule,
    matching: np.trapz(y=diff, x=p)

    No masking; uses all rows. Sorts by p to avoid signed/ordering issues.
    """
    res = np.asarray(res)
    p = np.asarray(p)
    if res.shape[0] != p.shape[0]:
        raise ValueError("`p` must have the same length as `res`.")

    order = np.argsort(p)
    y = res["distance"][order].astype(float, copy=False)
    x = p[order].astype(float, copy=False)

    return np.trapz(y=y, x=x)
