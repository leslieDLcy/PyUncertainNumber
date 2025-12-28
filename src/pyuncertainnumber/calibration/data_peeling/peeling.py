import numpy as numpy
from numpy.typing import NDArray
from .scenario import minimal_enclosing_hyperbox, is_inside_box, epsLU
from .fuzzy import boxes_to_fuzzy_projection
from scipy import stats

"""  Hints on notation used in this module:

### Forward problem ###
x: data set, each sample is a (nxd) vector.
d: dimension of each vector.
n: number of samples.

----
a: list (of length l) of subindices corresponding to the support vectors.
b: list (of length l) of enclosing sets.
l: number of levels.
p: violation upper probabilities, or membership values.
f: array (lxdx2) of projections of enclosing sets, for marginal plots.

### Backward problem ###
y: data set, each sample is a (nxd) vector.
d: dimension of each vector in the data set, or dimension of the output space.
n: number of samples.
`f`: a model or function such that `y=f(x)`.
`ux`: array (mxd_) of coverage samples.
m: number of coverage samples.
d_: dimension of input space.
uy: array (mxd) of evaluated coverage samples, such that `uy=f(ux)`.

----
a: list (of length l) of subindices corresponding to the support vectors.
b: list (of length l) of enclosing sets.
l: number of levels.
c: list (of length l) of subindices of coverage samples belonging to each level.
p: list (of length l) of violation upper probabilities, or membership values.
fx: array (lxd_x2) of projections of enclosing sets input space.
fy: array (lxdx2) of projections of enclosing sets output space.
"""


def data_peeling_algorithm(
    X: NDArray, tol: float = 1e-4
) -> tuple[list, list]:  # should also input the shape, rectangle, circle, etc.
    """Data peeling algorithm for constructing a sequence of nested enclosing sets.

    args:
        X (NDArray): data set of iid observations
        tol (float): a tolerance parameter determining the minimal size of an allowed enclosing box

    returns:
        tuple[list, list]: a tuple containing:
            - sequence_of_indices (list):
                a list (number of levels long) of sets (or list) of indices of heterogeneous size
            - sequence_of_boxes (list):
                a list (number of levels long) of (dx2) boxes


    .. figure:: /_static/peeling_illus.png
        :alt: ks_bounds
        :align: center
        :width: 90%

        Illustration of the data peeling algorithm.
    """
    H = make_hashtable(X)  # turns dataset in to hashtable
    n, d = X.shape
    sequence_of_indices, sequence_of_boxes = [], []
    index_num = 0  # number of support vectors per level
    j = 0  # number of levels
    while index_num < n:
        if j == 0:
            active_indices, box = minimal_enclosing_hyperbox(
                X
            )  # solve the optimization
            active_mask = index_to_mask(active_indices, n=n)
            nonactive_mask = active_mask == False
            X_strip = X[active_mask, :].copy()
            collect_indices = []
            for xstrip in X_strip:
                collect_indices.append(H[tuple(xstrip)])
                index_num += 1
            X_new = X[
                nonactive_mask, :
            ].copy()  # strip active scenarios from the original dataset
        else:
            active_indices, box = minimal_enclosing_hyperbox(X_new)
            active_mask = index_to_mask(active_indices, n=X_new.shape[0])
            nonactive_mask = active_mask == False
            X_strip = X_new[active_mask, :].copy()
            X_new = X_new[nonactive_mask, :].copy()
            collect_indices = []
            for xstrip in X_strip:
                collect_indices.append(H[tuple(xstrip)])
                index_num += 1
        sequence_of_indices.append(collect_indices)
        sequence_of_boxes.append(box)
        j += 1
    # return sequence_of_indices, sequence_of_boxes
    a, b = sanity_check(
        sequence_of_indices, sequence_of_boxes, tol=tol
    )  # this checks that the algorithm terminates as it is supposed to.
    return a, b


def index_to_mask(x: list, n=100) -> numpy.ndarray:
    """Converts the list of indices x into a (nx1) boolean mask, true at those particular indices

    Args:
        x (list): list of indices
        n (int, optional): size of the mask. Defaults to 100.

    Returns:
        numpy.ndarray: boolean mask
    """
    nrange = numpy.arange(n)
    boole = numpy.zeros(nrange.shape, dtype=bool)
    for xi in x:
        boole += nrange == xi
    return boole


def make_hashtable(x):
    h = {}
    for i, xi in enumerate(x):
        h[tuple(xi)] = i
    return h


def sanity_check(
    a, b, tol=1e-4
):  # checks that the smallest set has enough observation to make a decsion.
    b_last = b[-1]  # last enclosing set in the sequence
    for interval in b_last:
        if width(interval) < tol:
            return a[:-1], b[:-1]  # returns sequences stripped of the last element
    return a, b


def data_peeling_backward(
    uy: NDArray, y: NDArray = None, boxes: list = None, tol=1e-4
) -> tuple[list, list, list]:
    """
    Args:
        uy (NDArray): (mxd) array of coverage samples output space.
        boxes (list): sequence of boxes, each box is a (dx2) array. Also iterable of interval objects.

    Returns:
        tuple[list, list, list]: a tuple containing:
            - a: list (number of levels long) of sets (or list) of indices (heterogeneous size).
            - b: list (number of levels long) of (dx2) boxes (array-like).
            - c: list (number of levels long) of indices (input space) conatained in each level.
    Note:
        There are two cases where the peeling algorithm must raise an exception, and they are both linked to the termination of the algorithm.
            - When the last enclosing set has less samples than the minimum number of support scenarios to determine that set.
            - When there are enough samples to determine the set but some of them are too close to eachother (even for just one dimension).

    While two may be linked to the problem of degeneracy, in this context, it may be best suited to refer to this case as a coverage problem.
    """
    a = None  # initialise in case peeling forward is not called
    if (boxes is None) & (y is not None):
        a, b = data_peeling_algorithm(y, tol=tol)
    contained_in_each_level = []
    for box in b:
        contained_in_each_level.append(is_inside_box(uy, box))  # from bottom to top
    return (
        a,
        b,
        contained_in_each_level,
    )  # indices of samples contained in each level from bottom to top


def extract_kn(a):
    kk = 0
    k_cum = []
    for subi in a:
        kk += len(subi)
        k_cum.append(kk)
    return k_cum, kk


def peeling_to_structure(
    a: list, b: list, kind: str = "scenario", beta: float = 0.01
) -> tuple[NDArray, list]:
    """Peeling to structure.

    Args:
        a (list):  (number of levels long) of sets (or list) of indices of heterogeneous size
        b (list):  (number of levels long) of (dx2) boxes (array-like)

    Returns:
        f: (lxdx2) array containing the projections of a joint fuzzy number.
        p: list[float] of length l

    Note:
        Note b may not be boxes, but spheres, ellipses, etc.

    .. tip::
        keep b for piping: peeling_to_structure(data_peeling_algorithm(x))
    """
    k_cum, n = extract_kn(a)
    ALPHA = []
    ALPHA_append = ALPHA.append
    for k in k_cum:
        if kind == "scenario":
            alphaL, alphaU = epsLU(k, n, beta)
        elif kind == "c-box":
            pass
        elif kind == "clopper-pearson":
            pass
        elif kind == "percentage_sets":
            pass
        elif kind == "uniform":
            pass
        ALPHA_append(alphaU)  # each line may take a while if n is large
    f, p = boxes_to_fuzzy_projection(b, p=ALPHA)
    return f, p


def uniform(lo, hi, N=100):
    lo = numpy.asarray(lo, dtype=float)
    hi = numpy.asarray(hi, dtype=float)
    if lo.size == 1:
        d = 1
    else:
        d = lo.shape[0]  # if d == 0: stats.uniform(loc=lo, scale=hi-lo).rvs((N,))
    return stats.uniform(loc=lo, scale=hi - lo).rvs((N, d))


def samples_to_structure(ux: NDArray, c: list):
    """
    Args:
        ux (NDArray): (mxd_) array.
        c (list): (number of levels long) subset of indices (input space) conatained in each level.

    Returns:
        fx: (lxd_x2) array containing the projections of a joint fuzzy number in the input space.
    """

    pass


def width(x: NDArray):
    """
    Args:
        x: An interval or interval iterable, i.e. an (dx2) array

    Returns:
        w: the width of the intervals
    """
    x = numpy.asarray(x, dtype=float)
    if len(x.shape) == 1:
        return x[1] - x[0]
    else:
        return x[:, 1] - x[:, 0]
