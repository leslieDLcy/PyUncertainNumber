from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class eCDF_bundle:
    """a handy tuple of eCDF function q and p"""

    quantiles: np.ndarray
    probabilities: np.ndarray
    # TODO plot ecdf not starting from 0

    @classmethod
    def from_sps_ecdf(cls, e):
        """utility to tranform sps.ecdf to eCDF_bundle"""
        return cls(e.cdf.quantiles, e.cdf.probabilities)

    def plot_bounds(self, other):
        """plot the lower and upper bounds"""
        return plot_two_eCDF_bundle(self, other)


def transform_eCDF_bundle(e):
    """utility to tranform sps.ecdf to eCDF_bundle"""
    return eCDF_bundle(e.cdf.quantiles, e.cdf.probabilities)


def pl_ecdf_bounds_2(q1, p1, q2, p2, ax=None, marker="+"):
    """plot the upper and lower bounding cdf functions with two sets of quantiles and probabilities"""
    if ax is None:
        fig, ax = plt.subplots()

    ax.step(q1, p1, marker=marker, c="g", where="post")
    ax.step(q2, p2, marker=marker, c="b", where="post")
    ax.plot([q1[0], q2[0]], [0, 0], c="b")
    ax.plot([q1[-1], q2[-1]], [1, 1], c="g")
    return ax


def plot_two_eCDF_bundle(cdf1, cdf2, ax=None, **kwargs):
    """plot upper and lower eCDF_bundle objects"""
    if ax is None:
        fig, ax = plt.subplots()
    q1, p1 = cdf1.quantiles, cdf1.probabilities
    q2, p2 = cdf2.quantiles, cdf2.probabilities
    return pl_ecdf_bounds_2(q1, p1, q2, p2, ax=ax, **kwargs)


def pl_ecdf_bounding_bundles(
    b_l: eCDF_bundle,
    b_r: eCDF_bundle,
    ax=None,
    legend=True,
    title=None,
    sig_level=None,
    bound_colors=None,
    label=None,
    alpha=None,
    linestyle=None,
    linewidth=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    def set_if_not_none(d, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                d[k] = v

    plot_bound_colors = bound_colors if bound_colors is not None else ["g", "b"]

    cdf_kwargs = {"drawstyle": "steps-post"}

    set_if_not_none(
        cdf_kwargs,
        label=label,
        linestyle=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )

    ax.plot(
        b_l.quantiles,
        b_l.probabilities,
        label=label if label is not None else f"KS condidence bands {sig_level}\% ",
        color=plot_bound_colors[0],
        **cdf_kwargs,
    )
    ax.plot(
        b_r.quantiles,
        b_r.probabilities,
        color=plot_bound_colors[1],
        **cdf_kwargs,
    )
    ax.plot(
        [b_l.quantiles[0], b_r.quantiles[0]],
        [0, 0],
        color=plot_bound_colors[1],
        **cdf_kwargs,
    )
    ax.plot(
        [b_l.quantiles[-1], b_r.quantiles[-1]],
        [1, 1],
        color=plot_bound_colors[0],
        **cdf_kwargs,
    )

    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend()


def ecdf(d):
    """return the quantile and probability of a ecdf

    note:
        Scott's version which leads to doubling the length of quantiles and probabilities
        to make it a step function
    """
    d = np.array(d)
    N = d.size
    pp = np.concatenate((np.arange(N), np.arange(1, N + 1))) / N
    dd = np.concatenate((d, d))
    dd.sort()
    pp.sort()
    return dd, pp


def get_ecdf(s, w=None, display=False) -> tuple:
    """compute the weighted ecdf from (precise) sample data

    args:
        s (array-like) : precise sample data
        w (array-like) : weights

    note:
        - Sudret eq.1

    return:
        ecdf in the form of a tuple of q and p
    """

    if w is None:
        # weights
        N = len(s)
        w = np.repeat(1 / N, N)
    else:
        w = np.array(w)

    # s, w = sorting(s, w)
    arr = np.stack((s, w), axis=1)
    arr = arr[np.argsort(arr[:, 0])]

    p = np.cumsum(arr[:, 1])

    # for box plotting
    q = np.insert(arr[:, 0], 0, arr[0, 0], axis=0)
    p = np.insert(p, 0, 0.0, axis=0)

    if display == True:
        fig, ax = plt.subplots()
        ax.step(q, p, marker="+", where="post")

    # return quantile and probabilities
    return q, p
