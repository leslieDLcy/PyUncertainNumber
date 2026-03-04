import numpy as np
from pyuncertainnumber.pba.pbox_abc import Pbox, Staircase


def ascloseas_bounds(u, r, output_type="pbox"):
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
            xs, ws=np.full(n, 1 / n), r=r
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


def wasserstein_w1_cdf_envelope(xs, ws, r, x_grid=None):
    """
    Compute pointwise CDF envelopes for Q such that W1(P,Q) <= r,
    where P = sum_i ws[i] * delta_{xs[i]} (1D discrete distribution).

    args:
        xs          : 1D array of support points for P or data samples
        ws          : 1D array of weights for P (same size as xs, nonnegative, sum to 1)
        r           : radius for Wasserstein-1 ball around P
        x_grid      : optional 1D array of x values to evaluate envelopes at; if None, use sorted unique xs

    returns:
      xg          : grid of x values (sorted)
      F           : CDF of P at xg
      G_upper     : upper envelope over Q at xg
      G_lower     : lower envelope over Q at xg
    """
    xs = np.asarray(xs, dtype=float)
    ws = np.asarray(ws, dtype=float)
    assert xs.ndim == 1 and ws.ndim == 1 and xs.size == ws.size
    assert np.all(ws >= 0)
    s = ws.sum()
    assert s > 0
    ws = ws / s
    r = float(abs(r))

    # Sort by support points
    order = np.argsort(xs)
    xs = xs[order]
    ws = ws[order]
    n = xs.size

    # Prefix sums: W[i] = sum_{0..i-1} ws, XW[i] = sum_{0..i-1} ws*xs
    W = np.zeros(n + 1)
    XW = np.zeros(n + 1)
    W[1:] = np.cumsum(ws)
    XW[1:] = np.cumsum(ws * xs)

    # # Grid of x values to evaluate
    # if x_grid is None:
    #     # Good default: evaluate at sorted unique support points
    #     xg = np.unique(xs)
    # else:
    #     xg = np.sort(np.asarray(x_grid, dtype=float))

    # Grid of x values to evaluate
    if x_grid is None:
        # Use support points + midpoints + small extension beyond support
        xs_u = np.unique(np.sort(xs))

        if xs_u.size > 1:
            mid = (xs_u[:-1] + xs_u[1:]) / 2
            margin = r * 2  # a tuning parameter for plotting purpose
            xg = np.sort(
                np.concatenate([[xs_u[0] - margin], xs_u, mid, [xs_u[-1] + margin]])
            )
        else:
            # Degenerate case: single atom
            margin = r
            xg = np.array([xs_u[0] - margin, xs_u[0], xs_u[0] + margin])

    else:
        xg = np.sort(np.asarray(x_grid, dtype=float))

    def block_cost_right(s_idx, t_idx, x):
        """
        Cost to move ALL mass from indices [s_idx, t_idx] (inclusive) to x,
        assuming xs[j] >= x (right side). Cost = sum ws[j] * (xs[j] - x).
        """
        if t_idx < s_idx:
            return 0.0
        w_sum = W[t_idx + 1] - W[s_idx]
        xw_sum = XW[t_idx + 1] - XW[s_idx]
        return xw_sum - x * w_sum

    def block_cost_left(t_idx, i_idx, x):
        """
        Cost to move ALL mass from indices [t_idx, i_idx] (inclusive) to just right of x,
        assuming xs[j] <= x (left side). Cost = sum ws[j] * (x - xs[j]).
        """
        if i_idx < t_idx:
            return 0.0
        w_sum = W[i_idx + 1] - W[t_idx]
        xw_sum = XW[i_idx + 1] - XW[t_idx]
        return x * w_sum - xw_sum

    F = np.empty_like(xg)
    G_upper = np.empty_like(xg)
    G_lower = np.empty_like(xg)

    for k, x in enumerate(xg):
        # Split index: i is first index with xs[i] > x
        i = np.searchsorted(xs, x, side="right")

        # CDF of P at x
        Fk = W[i]
        F[k] = Fk

        # -------- Upper envelope: pull mass from RIGHT (indices i..n-1) to x --------
        # Want maximum moved mass under cost <= r using closest right points first.
        budget = r
        moved = 0.0

        if i < n and budget > 0:
            # Binary search for farthest t >= i such that moving ALL mass i..t fits budget
            lo, hi = i - 1, n - 1  # invariant: lo feasible, hi maybe not
            while lo < hi:
                mid = (lo + hi + 1) // 2
                c = block_cost_right(i, mid, x)
                if c <= budget + 1e-15:
                    lo = mid
                else:
                    hi = mid - 1
            t = lo

            if t >= i:
                # take full block i..t
                moved_full = W[t + 1] - W[i]
                cost_full = block_cost_right(i, t, x)
                moved += moved_full
                budget -= cost_full

            # take fractional from next atom (t+1) if any
            j = t + 1
            if j < n and budget > 0:
                d = xs[j] - x
                if d > 0:
                    take = min(ws[j], budget / d)
                    moved += take
                    budget -= take * d
                else:
                    # If xs[j] == x (shouldn't happen with side="right"), treat as free
                    moved += ws[j]

        G_upper[k] = min(1.0, Fk + moved)

        # -------- Lower envelope: push mass from LEFT (indices 0..i-1) to right of x --------
        # Max mass removable under cost <= r using closest left points first (largest xs first).
        budget = r
        moved_out = 0.0

        if i > 0 and budget > 0:
            # We take from the right end of the left side: indices ... i-1, i-2, ...
            # It's convenient to binary search on the LEFT boundary t (smallest index taken)
            # for taking the whole block t..(i-1).
            lo, hi = 0, i  # we'll search t in [0, i]
            # We want smallest t such that cost(t..i-1) <= budget; closer-only means larger t
            # We'll instead find largest t (closest block) that fits: t in [0..i-1]
            lo, hi = 0, i - 1
            best_t = i  # means "take nothing"
            # We'll binary search for the largest t such that moving all mass t..i-1 fits
            # Note cost decreases as t increases (block gets smaller & closer to x).
            left_lo, left_hi = 0, i - 1
            feasible_t = i  # default none
            while left_lo <= left_hi:
                mid = (left_lo + left_hi) // 2
                c = block_cost_left(mid, i - 1, x)
                if c <= budget + 1e-15:
                    feasible_t = mid
                    # try take more (make block bigger) => decrease mid
                    left_hi = mid - 1
                else:
                    left_lo = mid + 1

            if feasible_t < i:
                # This feasible_t is the SMALLEST t that fits, i.e. biggest block that fits.
                # Take full mass from feasible_t..i-1
                moved_full = W[i] - W[feasible_t]
                cost_full = block_cost_left(feasible_t, i - 1, x)
                moved_out += moved_full
                budget -= cost_full

                # If feasible_t > 0, we can take fraction from (feasible_t-1) (next farther left)
                j = feasible_t - 1
                if j >= 0 and budget > 0:
                    d = x - xs[j]
                    if d > 0:
                        take = min(ws[j], budget / d)
                        moved_out += take
                        budget -= take * d
                    else:
                        # xs[j] == x would be "free" to push just right in ideal math;
                        # in practice, treat with epsilon if needed.
                        moved_out += ws[j]

        G_lower[k] = max(0.0, Fk - moved_out)

    return xg, F, G_upper, G_lower
