import numpy as np
from pyuncertainnumber.pba.pbox_abc import Pbox, Staircase


def ascloseas_bounds(u, r, tail_tol=0.01, output_type="pbox"):
    """Get the bounds of all possible distributions Q such that W1(u, Q) <= r, where u is the base distribution.

    args:
        u: a distribution (e.g. an UncertainNumber)
        r: the radius of the Wasserstein-1 ball around u

    returns:
        envelope function that takes x and returns (lower_bound, upper_bound) for CDF of any Q with W1(u, Q) <= r
    """
    from pyuncertainnumber.pba.ecdf import eCDF_bundle
    from pyuncertainnumber.pba.pbox_abc import pbox_from_ecdf_bundle

    def helper_get_boounds(left_or_right):
        n = u.steps
        xs = u.left if left_or_right == "left" else u.right
        xg, F, G_upper, G_lower = wasserstein_w1_cdf_envelope(
            xs, ws=np.full(n, 1 / n), r=r, tail_tol=tail_tol
        )
        b_l = eCDF_bundle(xg, G_upper)
        b_r = eCDF_bundle(xg, G_lower)
        return b_l, b_r

    if u.degenerate:
        b_l, b_r = helper_get_boounds("left")
    else:
        b_l, _ = helper_get_boounds("left")
        _, b_r = helper_get_boounds("right")

    if output_type == "bounds":
        return b_l, b_r
    elif output_type == "pbox":
        return pbox_from_ecdf_bundle(b_l, b_r)
    else:
        raise ValueError(f"Invalid output_type: {output_type}")


def ascloseas_scott(B: Pbox, A: float):
    """Bound on all distributions that have area metric with B as small as or smaller than A.


    note:
        Scott's version of the implementation.
    """

    def areametric(u, v):
        return np.sum(np.abs(u - v)) / len(u)

    # if not is_pbox(B):
    #     B = Pbox(B)

    # u is left whle d is right edge.
    u, d = B.left.copy(), B.right.copy()
    n = len(u)
    a = A * n
    vu = np.empty(n, dtype=float)
    for i in range(n):
        s = np.sum(u[: i + 1])
        x = (s - a) / (i + 1)
        vu[i] = x
    for i in range(n):
        for j in range(i + 1, n):
            w = u.copy()
            w[i:j] = w[i]
            m = areametric(u, w)
            if A <= m:
                vu[j] = max(vu[j], u[i])
                break
    vd = np.empty(n, dtype=float)
    for i in range(n):
        tail = d[i:]  # work on the tail i..n-1
        m = len(tail)
        s = np.sum(tail)
        x = (s + a) / m  # sign flips because we push right
        vd[i] = x
    for i in range(n):
        for j in range(i + 1, n):
            w = d.copy()
            w[i:j] = w[i]  # flatten the suffix segment
            m = areametric(d, w)
            if A <= m:
                vd[i] = min(vd[i], d[j])
                break
    vu = np.maximum.accumulate(vu)  # enforce monotonicity (nondecreasing)
    vd = np.maximum.accumulate(vd)  # enforce monotonicity (nondecreasing)
    return Staircase(vu, vd)


def ascloseas_left(b: Pbox, a):
    """Bounds on all distributions that have area metric with b as small as or smaller than a.

    note:
        Scott's version of the implementation.
    """

    if not isinstance(b, Pbox):
        # b = Pbox(b)
        raise ValueError("b must be a Pbox for now")

    # u is left whle d is right edge.
    u, d = b.left.copy(), b.right.copy()
    n = len(u)
    A = a * n
    vu = np.empty(n, dtype=float)
    for i in range(n):
        s = np.sum(u[: i + 1])
        x = (s - A) / (i + 1)
        x = min(x, np.min(u[: i + 1]))
        vu[i] = x
    # vu = np.maximum.accumulate(vu)       # enforce monotonicity (nondecreasing)
    vd = np.empty(n, dtype=float)
    for i in range(n):
        tail = d[i:]  # work on the tail i..n-1
        m = len(tail)
        s = np.sum(tail)
        x = (s + A) / m  # sign flips because we push right
        x = max(x, np.max(tail))  # deviations nonnegative: x >= max(d[i..n-1])
        vd[i] = x
    # vd = np.maximum.accumulate(vd)       # enforce monotonicity (nondecreasing)
    return Staircase(left=vu, right=vd)


import numpy as np


def wasserstein_w1_cdf_envelope(xs, ws, r, x_grid=None, tail_tol=0.01, n_grid=500):
    """
    Compute pointwise CDF envelopes for Q such that W1(P,Q) <= r,
    where P = sum_i ws[i] * delta_{xs[i]} (1D discrete distribution).

    The default grid automatically extends far enough into the tails so that
    the envelopes are within 'tail_tol' of 0 (left) and 1 (right).

    Args:
        xs       : support points or data samples
        ws       : weights (same length as xs)
        r        : Wasserstein radius
        x_grid   : optional grid
        tail_tol : tolerance controlling how close the tails are to 0 and 1
        n_grid   : grid size if x_grid is None

    Returns:
        xg, F, G_upper, G_lower
    """

    xs = np.asarray(xs, dtype=float)
    ws = np.asarray(ws, dtype=float)

    if xs.ndim != 1 or ws.ndim != 1 or xs.size != ws.size:
        raise ValueError("xs and ws must be 1D arrays of the same length")

    if np.any(ws < 0):
        raise ValueError("weights must be nonnegative")

    ws = ws / ws.sum()
    r = abs(r)

    # sort support
    order = np.argsort(xs)
    xs = xs[order]
    ws = ws[order]

    n = xs.size

    # prefix sums
    W = np.zeros(n + 1)
    XW = np.zeros(n + 1)
    W[1:] = np.cumsum(ws)
    XW[1:] = np.cumsum(ws * xs)

    # ---- build default grid ----
    if x_grid is None:

        xmin = xs.min()
        xmax = xs.max()

        L = xmin - r / tail_tol
        U = xmax + r / tail_tol

        xg = np.linspace(L, U, n_grid)

    else:
        xg = np.sort(np.asarray(x_grid, dtype=float))

    # ---- transport cost helpers ----
    def block_cost_right(s_idx, t_idx, x):
        if t_idx < s_idx:
            return 0.0
        w_sum = W[t_idx + 1] - W[s_idx]
        xw_sum = XW[t_idx + 1] - XW[s_idx]
        return xw_sum - x * w_sum

    def block_cost_left(t_idx, i_idx, x):
        if i_idx < t_idx:
            return 0.0
        w_sum = W[i_idx + 1] - W[t_idx]
        xw_sum = XW[i_idx + 1] - XW[t_idx]
        return x * w_sum - xw_sum

    # ---- allocate output ----
    F = np.empty_like(xg)
    G_upper = np.empty_like(xg)
    G_lower = np.empty_like(xg)

    for k, x in enumerate(xg):

        i = np.searchsorted(xs, x, side="right")

        Fk = W[i]
        F[k] = Fk

        # -------- upper envelope --------
        budget = r
        moved = 0.0

        if i < n and budget > 0:

            lo, hi = i - 1, n - 1

            while lo < hi:
                mid = (lo + hi + 1) // 2
                if block_cost_right(i, mid, x) <= budget:
                    lo = mid
                else:
                    hi = mid - 1

            t = lo

            if t >= i:
                moved += W[t + 1] - W[i]
                budget -= block_cost_right(i, t, x)

            j = t + 1
            if j < n and budget > 0:
                d = xs[j] - x
                if d > 0:
                    moved += min(ws[j], budget / d)

        G_upper[k] = min(1.0, Fk + moved)

        # -------- lower envelope --------
        budget = r
        moved_out = 0.0

        if i > 0 and budget > 0:

            left_lo, left_hi = 0, i - 1
            feasible_t = i

            while left_lo <= left_hi:
                mid = (left_lo + left_hi) // 2
                if block_cost_left(mid, i - 1, x) <= budget:
                    feasible_t = mid
                    left_hi = mid - 1
                else:
                    left_lo = mid + 1

            if feasible_t < i:
                moved_out += W[i] - W[feasible_t]
                budget -= block_cost_left(feasible_t, i - 1, x)

                j = feasible_t - 1
                if j >= 0 and budget > 0:
                    d = x - xs[j]
                    if d > 0:
                        moved_out += min(ws[j], budget / d)
                    else:
                        moved_out += ws[j]

        G_lower[k] = max(0.0, Fk - moved_out)

    # enforce monotonicity
    F = np.maximum.accumulate(F)
    G_upper = np.maximum.accumulate(G_upper)
    G_lower = np.maximum.accumulate(G_lower)

    return xg, F, G_upper, G_lower
