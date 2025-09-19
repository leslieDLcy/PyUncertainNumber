from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .intervals.intervalOperators import wc_scalar_interval, make_vec_interval
from .intervals.number import Interval
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.legend_handler import HandlerBase
import sys
import numpy as np
from scipy.optimize import linprog


def inspect_un(x):
    """Inspect the any type of uncertain number x."""
    print(x.__repr__())
    x.display()


def extend_ecdf(cdf):
    """add zero and one to the ecdf

    args:
        CDF_bundle
    """
    if cdf.probabilities[0] != 0:
        cdf.probabilities = np.insert(cdf.probabilities, 0, 0)
        cdf.quantiles = np.insert(cdf.quantiles, 0, cdf.quantiles[0])
    if cdf.probabilities[-1] != 1:
        cdf.probabilities = np.append(cdf.probabilities, 1)
        cdf.quantiles = np.append(cdf.quantiles, cdf.quantiles[-1])
    return cdf


def sorting(list1, list2):
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return list1, list2


def reweighting(*masses):
    """reweight the masses to sum to 1"""
    masses = np.ravel(masses)
    return masses / masses.sum()


def uniform_reparameterisation(a, b):
    """reparameterise the uniform distribution to a, b"""
    #! incorrect in the case of Interval args
    a, b = wc_scalar_interval(a), wc_scalar_interval(b)
    return a, b - a


# TODO to test this high-performance version below
def find_nearest(array, value):
    """Find index/indices of nearest value(s) in `array` to each `value`.

    Efficient for both scalar and array inputs.
    """
    array = np.asarray(array)
    value_arr = np.atleast_1d(value)

    # Compute distances using broadcasting
    diff = np.abs(array[None, :] - value_arr[:, None])

    # Find index of minimum difference along axis 1
    indices = np.argmin(diff, axis=1)

    # Return scalar if input was scalar
    return indices[0] if np.isscalar(value) else indices


@mpl.rc_context({"text.usetex": True})
def plot_intervals(vec_interval: list[Interval], ax=None, **kwargs):
    """plot the intervals in a vectorised form
    args:
        vec_interval: vectorised interval objects
    """
    vec_interval = make_vec_interval(vec_interval)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    for i, intl in enumerate(vec_interval):  # horizontally plot the interval
        ax.plot([intl.lo, intl.hi], [i, i], **kwargs)
    ax.margins(x=0.1, y=0.1)
    ax.set_yticks([])
    return ax


def read_json(file_name):
    f = open(file_name)
    data = json.load(f)
    return data


def is_increasing(arr):
    """check if 'arr' is increasing"""
    return np.all(np.diff(arr) >= 0)


class NotIncreasingError(Exception):
    pass


# TODO: integrate the two sub-functions to make more consistent.
def condensation(bound, number: int):
    """a joint implementation for condensation

    args:
        number (int) : the number to be reduced
        bound (array-like): either the left or right bound to be reduced

    note:
        It will keep the first and last from the bound
    """

    if isinstance(bound, list | tuple):
        return condensation_bounds(bound, number)
    else:
        return condensation_bound(bound, number)


def condensation_bounds(bounds, number):
    """condense the bounds of number pbox

    args:
        number (int) : the number to be reduced
        bounds (list or tuple): the left and right bound to be reduced
    """
    b = bounds[0]

    if number > len(b):
        raise ValueError("Cannot sample more elements than exist in the list.")
    if len(bounds[0]) != len(bounds[1]):
        raise Exception("steps of two bounds are different")

    indices = np.linspace(0, len(b) - 1, number, dtype=int)

    l = np.array([bounds[0][i] for i in indices])
    r = np.array([bounds[1][i] for i in indices])
    return l, r


def condensation_bound(bound, number):
    """condense the bounds of number pbox

    args:
        number (int) : the number to be reduced
        bound (array-like): either the left or right bound to be reduced
    """

    if number > len(bound):
        raise ValueError("Cannot sample more elements than exist in the list.")

    indices = np.linspace(0, len(bound) - 1, number, dtype=int)

    new_bound = np.array([bound[i] for i in indices])
    return new_bound


def smooth_condensation(bounds, number=200):

    def smooth_ecdf(V, steps):

        m = len(V) - 1

        if m == 0:
            return np.repeat(V, steps)
        if steps == 1:
            return np.array([min(V), max(V)])

        d = 1 / m
        n = round(d * steps * 200)

        if n == 0:
            c = V
        else:
            c = []
            for i in range(m):
                v = V[i]
                w = V[i + 1]
                c.extend(np.linspace(start=v, stop=w, num=n))

        u = [c[round((len(c) - 1) * (k + 0) / (steps - 1))] for k in range(steps)]

        return np.array(u)

    l_smooth = smooth_ecdf(bounds[0], number)
    r_smooth = smooth_ecdf(bounds[1], number)
    return l_smooth, r_smooth


def equi_selection(arr, n):
    """draw n equidistant points from the array"""
    indices = np.linspace(0, len(arr) - 1, n, dtype=int)
    selected = arr[indices]
    return selected


# --- Reuse pbox rectangle key function ---
def create_colored_edge_box(x0, y0, width, height, linewidth=1):
    verts_top = [(x0, y0 + height), (x0 + width, y0 + height)]
    verts_left = [(x0, y0), (x0, y0 + height)]
    verts_bottom = [(x0, y0), (x0 + width, y0)]
    verts_right = [(x0 + width, y0), (x0 + width, y0 + height)]

    def make_patch(verts, color):
        path = mpath.Path(verts)
        return mpatches.PathPatch(
            path, edgecolor=color, facecolor="none", linewidth=linewidth
        )

    return [
        make_patch(verts_top, "green"),
        make_patch(verts_left, "green"),
        make_patch(verts_bottom, "blue"),
        make_patch(verts_right, "blue"),
    ]


# --- Custom pbox legend handler ---
class CustomEdgeRectHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        rect_patches = create_colored_edge_box(
            xdescent, ydescent, width, height, linewidth=1
        )
        for patch in rect_patches:
            patch.set_transform(trans)
        return rect_patches


def expose_functions_as_public(mapping, wrapper):
    """expose private functions as public APIs

    args:
        mapping (dict): a dictionary containing private function names mapped to public APIs
        wrapper (callable): a function that wraps the original functions (e.g., the decorator UNtoUN)

    note:
        the decorator which wraps the original function returning Pbox into returning UN, hence making the public UN API
    """
    # Get the module that called this function
    caller_globals = sys._getframe(1).f_globals
    for name, fn in mapping.items():
        caller_globals[name] = wrapper(fn)


def left_right_switch(left, right):
    """
    note:
        right quantile should be greater and equal than left quantile
    """
    if np.all(left >= right):
        # If left is greater than right, switch them
        left, right = right, left
        return left, right
    else:
        return left, right


# * ----------------------- pbox variance bounds via LP -----------------------*#


def build_constraints_from_pbox(q_a, p_a, q_b, p_b, x_grid, n=200, eps=1e-12):
    x = np.asarray(x_grid, float)
    p_a = np.asarray(p_a, float)
    q_a = np.asarray(q_a, float)
    p_b = np.asarray(p_b, float)
    q_b = np.asarray(q_b, float)

    # envelopes on x-grid
    def step_cdf_from_quantile(q, p, xg):
        idx = np.searchsorted(q, xg, side="right") - 1
        F = np.where(idx >= 0, p[np.clip(idx, 0, len(p) - 1)], 0.0)
        return np.clip(np.maximum.accumulate(F), 0.0, 1.0)

    F_L = step_cdf_from_quantile(q_b, p_b, x)  # lower CDF uses upper quantile curve
    F_U = step_cdf_from_quantile(q_a, p_a, x)  # upper CDF uses lower quantile curve
    F_L = np.minimum(F_L, F_U)
    # integer bounds with tolerance; convert to probabilities
    L = np.maximum.accumulate(np.ceil(n * (F_L - eps)).astype(int))
    U = np.maximum.accumulate(np.floor(n * (F_U + eps)).astype(int))
    U[-1] = n
    return x, L, U, n


def lp_mean_bounds(x, L, U, n):
    m = len(x)
    # Variables p_i >= 0
    A_ub = []
    b_ub = []
    # cumulative upper: sum_{j<=i} p_j <= U[i]/n
    for i in range(m):
        row = np.zeros(m)
        row[: i + 1] = 1.0
        A_ub.append(row)
        b_ub.append(U[i] / n)
    # cumulative lower: -sum_{j<=i} p_j <= -L[i]/n
    for i in range(m):
        row = np.zeros(m)
        row[: i + 1] = -1.0
        A_ub.append(row)
        b_ub.append(-L[i] / n)
    # equality sum p_i = 1 → as two inequalities (linprog has A_eq too if you prefer)
    A_eq = [np.ones(m)]
    b_eq = [1.0]
    bounds = [(0, 1) for _ in range(m)]
    # μmin
    res1 = linprog(
        c=x, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    # μmax
    res2 = linprog(
        c=-x, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if not (res1.success and res2.success):
        raise RuntimeError("Mean LP infeasible.")
    return res1.fun, -res2.fun, A_ub, b_ub, A_eq, b_eq, bounds


def lp_E2_at_mean(x, A_ub, b_ub, A_eq, b_eq, bounds, mu, maximize=True):
    # add equality for mean: sum p_i x_i = mu
    Aeq = A_eq + [x]
    beq = b_eq + [mu]
    c = -(x**2) if maximize else (x**2)
    res = linprog(
        c=c, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs"
    )
    if not res.success:
        return None
    E2 = -res.fun if maximize else res.fun
    return E2


def variance_bounds_via_lp(q_a, p_a, q_b, p_b, x_grid, n=None, mu_grid=101):
    """Approximate variance bounds of a p-box via linear programming.

    It is based on discretisation.

    args:
        q_a, p_a: lower quantile and its probabilities (upper CDF bound)
        q_b, p_b: upper quantile and its probabilities (lower CDF bound)
        x_grid: grid of x values to build constraints on; e.g., np.linspace(xmin, xmax, 200)
        n: number of discrete points to represent the p-box; if None, use len(p_a)
        mu_grid: number of mean values to scan between μmin and μmax
    """
    if n is None:
        n = len(p_a)
    x, L, U, n = build_constraints_from_pbox(q_a, p_a, q_b, p_b, x_grid, n=n)
    mu_min, mu_max, A_ub, b_ub, A_eq, b_eq, bounds = lp_mean_bounds(x, L, U, n)
    mus = np.linspace(mu_min, mu_max, mu_grid)
    vmin = np.inf
    vmax = -np.inf
    qmin_mu = qmax_mu = None
    for mu in mus:
        E2_max = lp_E2_at_mean(x, A_ub, b_ub, A_eq, b_eq, bounds, mu, maximize=True)
        E2_min = lp_E2_at_mean(x, A_ub, b_ub, A_eq, b_eq, bounds, mu, maximize=False)
        if E2_max is not None:
            vmax = max(vmax, E2_max - mu**2)
            qmax_mu = mu if vmax == E2_max - mu**2 else qmax_mu
        if E2_min is not None:
            vmin = min(vmin, E2_min - mu**2)
            qmin_mu = mu if vmin == E2_min - mu**2 else qmin_mu
    return dict(var_min=vmin, var_max=vmax, mu_min=mu_min, mu_max=mu_max)


def get_mean_var_from_ecdf(q, p):
    """Numerically estimate the mean and var from ECDF data

    args:
        q (array-like): quantiles
        p (array-like): probabilities

    example:
        >>> # Given ECDF data an example
        >>> q = [1, 2, 3, 4]
        >>> p = [0.25, 0.5, 0.75, 1.0]
        >>> mean, var = get_mean_var_from_ecdf(q, p)
    """

    # Step 1: Recover PMF
    pmf = [p[0]] + [p[i] - p[i - 1] for i in range(1, len(p))]

    # Step 2: Compute Mean
    mean = sum(x * p for x, p in zip(q, pmf))

    # Step 3: Compute Variance
    variance = sum(p * (x - mean) ** 2 for x, p in zip(q, pmf))
    return mean, variance


import numpy as np


def sample_ecdf_in_pbox(q_a, p_a, q_b, p_b, x_grid=None, n=None, rng=None, eps=1e-12):
    """
    Sample a random ECDF (quantile & probability vectors) lying inside the p-box
    defined by lower envelope (q_a, p_a) and upper envelope (q_b, p_b).

    args:
        q_a, p_a : arrays
            Lower (left) bounding quantile function sampled at probabilities p_a.
        q_b, p_b : arrays
            Upper (right) bounding quantile function sampled at probabilities p_b.
        x_grid : array or None
            Discrete support where ECDF masses are allowed to sit. If None, uses the
            union of q_a and q_b sorted and uniqued. You can also pass a custom grid
            (e.g., np.linspace(min(q_a), max(q_b), 200)).
        n : int or None
            ECDF size (number of jumps). If None, defaults to len(p_a).
        rng : numpy.random.Generator or None
            Random generator to control reproducibility.
        eps : float
            Small tolerance to avoid rounding issues in DP bounds.

    Returns
        q : ndarray (length n)
            Quantile vector (nondecreasing), values taken from x_grid.
        p : ndarray (length n)
            Probability vector for the ECDF: p[r] = (r+1)/n.

    note:
        Choose an x-grid (or let the function use the union of q_a and q_b).
        # x_grid = np.linspace(min(q_a), max(q_b), 200)  # e.g., a uniform 200-point support

    example:
        >>> p = pba.normal([4, 6], 1)
        >>> ecdf_q, ecdf_p = sample_ecdf_in_pbox(p.left, p.p_values, p.right, p.p_values)
    """

    # --- helpers ---
    def _step_cdf_from_quantile(q, p, x):
        """Right-continuous step CDF F(x) = sup{ p_j : q_j <= x } on grid x."""
        q = np.asarray(q, float)
        p = np.asarray(p, float)
        x = np.asarray(x, float)
        idx = np.searchsorted(q, x, side="right") - 1
        F = np.where(idx >= 0, p[np.clip(idx, 0, len(p) - 1)], 0.0)
        F = np.maximum.accumulate(F)
        return np.clip(F, 0.0, 1.0)

    def _pbox_cdf_bounds_on_grid(q_a, p_a, q_b, p_b, x):
        """Lower/upper CDF envelopes on x: F_L (using q_b,p_b), F_U (using q_a,p_a)."""
        F_L = _step_cdf_from_quantile(q_b, p_b, x)  # lower envelope via upper quantiles
        F_U = _step_cdf_from_quantile(q_a, p_a, x)  # upper envelope via lower quantiles
        F_L = np.minimum(F_L, F_U)
        return F_L, F_U

    def _dp_count_robust(F_L, F_U, n, eps):
        """Robust DP with tolerant rounding + endpoint repairs."""
        F_L = np.asarray(F_L, float)
        F_U = np.asarray(F_U, float)
        m = len(F_L)
        # integer cumulative bounds for counts
        L = np.ceil(n * (F_L - eps)).astype(int)
        U = np.floor(n * (F_U + eps)).astype(int)
        L = np.clip(np.maximum.accumulate(L), 0, n)
        U = np.clip(np.maximum.accumulate(U), 0, n)
        # enforce final endpoint can reach n
        U[-1] = n
        if np.any(L > U):
            return 0, np.zeros((m, n + 1), dtype=object), L, U
        DP = np.zeros((m, n + 1), dtype=object)
        if L[0] <= U[0]:
            DP[0, L[0] : U[0] + 1] = 1
        for i in range(1, m):
            pref = np.cumsum(DP[i - 1])
            lo, hi = L[i], U[i]
            if lo <= hi:
                DP[i, lo : hi + 1] = pref[lo : hi + 1]
        total = int(DP[m - 1, n])
        return total, DP, L, U

    def _sample_counts(DP, L, U, n, rng):
        """Sample a feasible cumulative path uniformly, return counts k on x_grid."""
        if rng is None:
            rng = np.random.default_rng()
        m = DP.shape[0]
        C = np.zeros(m, dtype=int)
        c = n
        for i in reversed(range(m)):
            if i == 0:
                if DP[0, c] == 0:
                    raise RuntimeError("Infeasible path at start.")
                C[0] = c
                break
            lo = L[i - 1]
            hi = min(U[i - 1], c)
            w = DP[i - 1, lo : hi + 1].astype(object)
            W = np.asarray(w, float)
            s = W.sum()
            if s <= 0:
                raise RuntimeError("Infeasible path while sampling.")
            probs = W / s
            t = rng.choice(np.arange(lo, hi + 1), p=probs)
            C[i] = c
            c = int(t)
        k = np.diff(np.concatenate(([0], C)))
        return k  # length m, sums to n

    # --- inputs & grid ---
    q_a = np.asarray(q_a, float)
    p_a = np.asarray(p_a, float)
    q_b = np.asarray(q_b, float)
    p_b = np.asarray(p_b, float)

    if n is None:
        n = len(p_a)
    if x_grid is None:
        # default: union of quantile support from both envelopes
        x_grid = np.unique(np.concatenate([q_a, q_b]))
    x = np.asarray(x_grid, float)

    # --- envelopes, DP, sampling ---
    F_L, F_U = _pbox_cdf_bounds_on_grid(q_a, p_a, q_b, p_b, x)
    total, DP, L, U = _dp_count_robust(F_L, F_U, n, eps=eps)
    if total == 0:
        raise ValueError(
            "No admissible ECDFs on the chosen x_grid with given envelopes."
        )
    k = _sample_counts(DP, L, U, n, rng)

    # --- build (q, p) for the sampled ECDF ---
    q = np.repeat(x, k)  # length n, nondecreasing
    p = (np.arange(1, n + 1)) / n  # standard ECDF probabilities
    return q, p
