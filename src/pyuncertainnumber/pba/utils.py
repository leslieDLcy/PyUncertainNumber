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
import numpy as np
from scipy.optimize import linprog


def build_constraints_from_pbox_robust(q_a, p_a, q_b, p_b, x_grid, n=None, eps=1e-12):
    """
    Returns x, L, U, n and also the envelopes F_L, F_U on x_grid.
    If F_L==F_U (degenerate p-box), we set L=U=round(n*F) exactly to avoid infeasibility.
    """
    x = np.asarray(x_grid, float)
    q_a = np.asarray(q_a, float)
    p_a = np.asarray(p_a, float)
    q_b = np.asarray(q_b, float)
    p_b = np.asarray(p_b, float)
    if n is None:
        n = len(p_a)

    def step_cdf_from_quantile(q, p, xg):
        idx = np.searchsorted(q, xg, side="right") - 1
        F = np.where(idx >= 0, p[np.clip(idx, 0, len(p) - 1)], 0.0)
        return np.clip(np.maximum.accumulate(F), 0.0, 1.0)

    # Envelopes on x-grid
    F_L = step_cdf_from_quantile(q_b, p_b, x)  # lower CDF via upper quantiles
    F_U = step_cdf_from_quantile(q_a, p_a, x)  # upper CDF via lower quantiles
    F_L = np.minimum(F_L, F_U)

    if np.allclose(F_L, F_U, atol=1e-12, rtol=0):
        # Degenerate case: one distribution. Build exact integer cumulatives.
        C = np.rint(n * F_U).astype(int)
        C = np.clip(np.maximum.accumulate(C), 0, n)
        C[-1] = n
        L = C.copy()
        U = C.copy()
    else:
        # General case with tolerant integer bounds
        L = np.ceil(n * (F_L - eps)).astype(int)
        U = np.floor(n * (F_U + eps)).astype(int)
        L = np.clip(np.maximum.accumulate(L), 0, n)
        U = np.clip(np.maximum.accumulate(U), 0, n)
        U[-1] = n
        # Safety
        if np.any(L > U):
            raise RuntimeError("Infeasible cumulative bounds (L>U) after rounding.")

    return x, L, U, n, F_L, F_U


def variance_bounds_via_lp(q_a, p_a, q_b, p_b, x_grid, n=None, mu_grid=101):
    x, L, U, n, F_L, F_U = build_constraints_from_pbox_robust(
        q_a, p_a, q_b, p_b, x_grid, n=n
    )

    # Degenerate p-box → single distribution
    if np.allclose(F_L, F_U, atol=1e-12, rtol=0):
        counts = np.diff(np.concatenate(([0], U)))  # because L==U==C
        probs = counts / n
        mu = float(np.dot(probs, x))
        E2 = float(np.dot(probs, x**2))
        var = E2 - mu**2
        return dict(var_min=var, var_max=var, mu_min=mu, mu_max=mu, probs_on_grid=probs)

    # General case (same as before): build LPs
    m = len(x)
    A_ub = []
    b_ub = []
    # cumulative uppers
    for i in range(m):
        row = np.zeros(m)
        row[: i + 1] = 1.0
        A_ub.append(row)
        b_ub.append(U[i] / n)
    # cumulative lowers
    for i in range(m):
        row = np.zeros(m)
        row[: i + 1] = -1.0
        A_ub.append(row)
        b_ub.append(-L[i] / n)
    A_eq = [np.ones(m)]
    b_eq = [1.0]
    bounds = [(0, 1) for _ in range(m)]

    # Mean bounds
    r1 = linprog(
        c=x, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    r2 = linprog(
        c=-x, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if not (r1.success and r2.success):
        raise RuntimeError("Mean LP infeasible.")
    mu_min, mu_max = r1.fun, -r2.fun

    # Sweep μ (or replace with a 1D optimizer if you like)
    mus = np.linspace(mu_min, mu_max, mu_grid)
    vmin, vmax = np.inf, -np.inf
    for mu in mus:
        # Add mean equality
        Aeq = A_eq + [x]
        beq = b_eq + [mu]
        # Max E[X^2]
        res_max = linprog(
            c=-(x**2),
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=Aeq,
            b_eq=beq,
            bounds=bounds,
            method="highs",
        )
        if res_max.success:
            vmax = max(vmax, -res_max.fun - mu**2)
        # Min E[X^2]
        res_min = linprog(
            c=(x**2),
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=Aeq,
            b_eq=beq,
            bounds=bounds,
            method="highs",
        )
        if res_min.success:
            vmin = min(vmin, res_min.fun - mu**2)

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


def area_between_ecdfs(x_upper, p_upper, x_lower, p_lower):
    """Compute the area between two ECDFs defined by (x_upper, p_upper) and (x_lower, p_lower).

    args:
        x_upper, p_upper: arrays defining the upper ECDF
        x_lower, p_lower: arrays defining the lower ECDF

    """
    # union grid of breakpoints
    grid = np.unique(np.concatenate([x_upper, x_lower]))
    if grid.size < 2:
        return 0.0  # degenerate case

    widths = np.diff(grid)  # interval widths [grid[k], grid[k+1])
    lefts = grid[:-1]  # left endpoints of intervals

    Fu = _ecdf_value_on_left_of_intervals(x_upper, p_upper, lefts)
    Fl = _ecdf_value_on_left_of_intervals(x_lower, p_lower, lefts)

    area = np.sum(widths * np.abs(Fu - Fl))
    return float(area)


def _ecdf_value_on_left_of_intervals(x, p, grid_left):
    """
    Right-continuous ECDF value used on [grid_left[k], grid_right[k]).
    For values below the first x, value is 0. For above the last x, value stays at p[-1].
    """
    # index of last x <= each grid_left (right-continuous step)
    idx = np.searchsorted(x, grid_left, side="right") - 1
    vals = np.where(idx >= 0, p[idx], 0.0)
    return vals
