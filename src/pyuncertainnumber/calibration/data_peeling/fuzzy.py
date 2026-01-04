import numpy
import numpy.matlib as matlib
import scipy.stats as stats
from numpy.typing import NDArray


def samples_to_fuzzy_projection(ux: NDArray, c: list) -> NDArray:
    """
    Args:
        ux (NDArray): an (mxd_) array of samples, usually uniform. m is a large integer.

        c (list): a list (of length l) of subindices of coverage samples belonging to each level.
            `len(levels) < m` must yield `True`, `sum([sum(len(subi)) for subi in levels])==m` must yield `True`.

    Returns:
        fx (NDArray): returns a d-dimensional fuzzy number, i.e. an (lxdx2) array.
    """
    ux = numpy.asarray(ux, dtype=float)
    m, d_ = ux.shape
    l = len(c)
    fx = numpy.empty((l, d_, 2))
    first_empty_level = l
    for i, subi in enumerate(c):  # levels should be arranged from bottom to top
        if sum(subi) == 0:
            first_empty_level = (
                i  # set (non-empty) starting level. Yes, there may be empty levels
            )
            print(
                f"levels {i}:{l} have no sample coverage. Consider increasing the sample size."
            )
    for i in range(
        l
    ):  # levels are usually a handful, so for loop does not slow things too much
        fx[i, :, 0] = numpy.min(ux[c[i], :], axis=0)
        fx[i, :, 1] = numpy.max(ux[c[i], :], axis=0)
    if first_empty_level < l:
        for i in range(first_empty_level, l):
            fx[i, :, 0] = numpy.min(ux[c[first_empty_level - 1], :], axis=0)
            fx[i, :, 1] = numpy.max(ux[c[first_empty_level - 1], :], axis=0)
    return fx


def samples_to_fuzzy_multivariate(u: numpy.ndarray, levels: list, p: list = None):
    pass


def boxes_to_fuzzy_projection(boxes: list, p: list = None):
    """
    Args:
        boxes (list): sequence of boxes, each box is a (dx2) array. Also iterable of interval objects. Second output of the forward data-peeling algorithm.

    Returns:
        f (NDArray): an (lxdx3) fuzzy projection data structure
    """
    l = len(boxes)
    d, _ = boxes[0].shape  # (dx2)
    if p is None:
        p = numpy.arange(1 / l, 1 + 1 / l, 1 / l)  # p = 1/l
    fuzzyall = numpy.empty((l, d, 2))
    fuzzyall[:, :, :2] = numpy.asarray(
        boxes, dtype=float
    )  # f[:,:,2] = matlib.repmat(p,d,1).T
    return fuzzyall, p


def coverage_samples(lo: NDArray, hi: NDArray, m: int = 1_000):
    """
    Args:
        lo (NDArray): an (d,) array (or list) of left endpoints. Coverage means samples are generated using low-discrepancy schemes.
        hi (NDArray): an (d,) array (or list) of right endpoints, with hi > lo.

    Returns:
        u (NDArray): an (mxd_) array of coverage samples
    """
    lo, hi = numpy.asarray(lo, dtype=float), numpy.asarray(hi, dtype=float)
    if lo.size == 1:
        d = 1
    else:
        d = lo.shape[0]
    dist = stats.uniform(loc=lo, scale=hi - lo)  # <- change sampling scheme here
    return dist.rvs((m, d))


def width(x: NDArray) -> NDArray:
    """
    Args:
        x (NDArray): an interval iterable, i.e. an (dx2) array

    Returns:
        w (NDArray): the width of the intervals
    """
    x = numpy.asarray(x, dtype=float)
    return x[:, 1] - x[:, 0]
