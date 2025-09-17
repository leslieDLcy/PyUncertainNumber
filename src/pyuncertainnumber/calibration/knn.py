
from calibration import *
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity


class KNNCalibrator(Calibrator):
    r"""
    Unified kNN-based calibrator for black-box models or precomputed simulations.

    Setup (unified)
    ---------------
    - If `evaluate_model=False` and `simulated_data` is provided, we **reuse** simulations and
      build a per-design kNN index by **filtering** rows with |xi - xi*| < a_tol for each ξ* in `xi_list`.
    - If `evaluate_model=True`, we **simulate** `y = model(theta, xi)` **for each** ξ in `xi_list`, using
      a **shared θ grid** drawn once from `theta_sampler(n_samples)`. Then we build per-design kNN indices.

    Calibration (single logic for single/multi-design)
    --------------------------------------------------
    For each `(y_obs, xi)`:
      1) standardize `y_obs` with the per-design scaler,
      2) get k nearest neighbors in y-space,
      3) map indices → θ for that design.
    Finally we **stack** θ across all observations/designs (or optionally tally/vote).

    Args
    ----
    knn : int
        Number of neighbors per observed row.
    a_tol : float
        Tolerance for matching `simulated_data['xi']` to a requested ξ* (when reusing).
    evaluate_model : bool
        If True, call the black-box `model` for each ξ in `xi_list` on a shared θ grid.
        If False, reuse `simulated_data` (requires y/theta/xi).
    random_state : Optional[int]
        Seed for reproducibility (affects theta_sampler and resampling).
    """

    def __init__(self,
                 knn: int = 100,
                 a_tol: float = 0.05,
                 evaluate_model: bool = False
                 ):

        super().__init__()
        self.knn = int(knn)
        self.a_tol = float(a_tol)
        self.evaluate_model = bool(evaluate_model)
        self.random_state = 42

        # Internal state
        self._theta_grid: Optional[np.ndarray] = None            # shared grid if evaluate_model=True, else unused
        self._theta_by_xi: Dict[Tuple[float, ...], np.ndarray] = {}  # per-design θ (may be shared ref)
        self._y_by_xi: Dict[Tuple[float, ...], np.ndarray] = {}      # per-design y
        self._scaler_by_xi: Dict[Tuple[float, ...], StandardScaler] = {}
        self._neigh_by_xi: Dict[Tuple[float, ...], NearestNeighbors] = {}
        self._grid_idx_by_xi: Dict[Tuple[float, ...], np.ndarray] = {}
        self._posterior: Optional[Dict[str, Any]] = None

        # Keep original sims if reusing
        self._sim_y: Optional[np.ndarray] = None
        self._sim_theta: Optional[np.ndarray] = None
        self._sim_xi: Optional[np.ndarray] = None


    # ---------- utilities ----------
    @staticmethod
    def _key_from_xi(xi) -> Tuple[float, ...]:
        """Stable tuple key for a scalar/vector design ξ."""
        return tuple(np.atleast_1d(np.asarray(xi, float)).ravel())

    # ---------- setup ----------
    def setup(self,
              model: Optional[Callable[[np.ndarray, Union[float, np.ndarray]], np.ndarray]] = None,
              theta_sampler: Optional[Callable[[int], np.ndarray]] = None,
              simulated_data: Optional[Dict[str, np.ndarray]] = None,
              xi_list: Optional[List[Union[float, np.ndarray]]] = None,
              n_samples: int = 10000):
        """
        Prepare per-design kNN structures by either reusing `simulated_data` or by simulating for each design.

        Parameters
        ----------
        model : callable
            Black-box simulator with signature `model(theta, xi) -> y` (vectorized over theta).
        theta_sampler : callable
            Sampler for θ; required when `evaluate_model=True`.
        simulated_data : dict
            Dict with keys {"y": (n, dy), "theta": (n, dθ), "xi": (n, dξ)} when reusing sims.
        xi_list : list
            List of designs; each item can be scalar or array-like. If None → [0.0].
        n_samples : int
            Number of θ samples to draw when `evaluate_model=True`.
        """
        xi_list = [0.0] if not xi_list else xi_list

        # Reset state
        self._theta_grid = None
        self._theta_by_xi.clear()
        self._y_by_xi.clear()
        self._scaler_by_xi.clear()
        self._neigh_by_xi.clear()
        self._posterior = None

        if not self.evaluate_model:
            # ---- Reuse provided simulations; filter per design ----
            if simulated_data is None:
                raise ValueError("evaluate_model=False requires `simulated_data` with keys 'y','theta','xi'.")

            self._sim_y = np.asarray(simulated_data["y"], float)
            self._sim_theta = np.asarray(simulated_data["theta"], float)
            self._sim_xi = np.asarray(simulated_data.get("xi", None), float)
            if self._sim_xi is None:
                raise ValueError("`simulated_data` must include 'xi' to filter per design.")

            for xi in xi_list:
                key = self._key_from_xi(xi)
                mask = np.all(np.abs(self._sim_xi - np.atleast_1d(xi)) < self.a_tol, axis=1)
                y_xi = self._sim_y[mask]
                theta_xi = self._sim_theta[mask]
                if y_xi.size == 0:
                    raise ValueError(f"No simulations matched design {xi} within tolerance a_tol={self.a_tol}.")
                # drop NaNs rows in y
                ok = ~np.isnan(y_xi).any(axis=1)
                y_xi, theta_xi = y_xi[ok], theta_xi[ok]
                if y_xi.size == 0:
                    raise ValueError(f"All simulations at design {xi} had NaNs in y.")
                # build scaler & kNN
                sc = StandardScaler().fit(y_xi)
                neigh = NearestNeighbors(n_neighbors=self.knn).fit(sc.transform(y_xi))
                # store
                self._theta_by_xi[key] = theta_xi
                self._y_by_xi[key] = y_xi
                self._scaler_by_xi[key] = sc
                self._neigh_by_xi[key] = neigh

        else:
            # ---- Evaluate model per design on a shared θ grid ----
            if model is None or theta_sampler is None:
                raise ValueError("evaluate_model=True requires `model` and `theta_sampler`.")
            self._theta_grid = np.asarray(theta_sampler(int(n_samples)), float)
            if self._theta_grid.ndim != 2:
                raise ValueError("theta_sampler must return a 2D array (n_samples, dθ).")

            for xi in xi_list:
                key = self._key_from_xi(xi)
                # vectorized model call over θ
                y_xi = np.asarray(model(self._theta_grid, xi), float)
                if y_xi.ndim == 1:
                    y_xi = y_xi[:, None]
                if y_xi.shape[0] != self._theta_grid.shape[0]:
                    raise ValueError("Model must return one row of y per θ sample.")
                # drop NaNs rows in y (and corresponding θ rows)
                ok = ~np.isnan(y_xi).any(axis=1)
                y_xi = y_xi[ok]
                theta_xi = self._theta_grid[ok]
                grid_idx = np.where(ok)[0]
                self._grid_idx_by_xi[key] = grid_idx  # maps local row j -> global grid index grid_idx[j]

                if y_xi.size == 0:
                    raise ValueError(f"All simulations at design {xi} had NaNs in y.")
                # build scaler & kNN
                sc = StandardScaler().fit(y_xi)
                neigh = NearestNeighbors(n_neighbors=self.knn).fit(sc.transform(y_xi))
                # store
                self._theta_by_xi[key] = theta_xi
                self._y_by_xi[key] = y_xi
                self._scaler_by_xi[key] = sc
                self._neigh_by_xi[key] = neigh

        self.is_ready = True

    # ---------- nearest ----------
    def nearest(self,
                y: Union[np.ndarray, List[float]],
                xi: Union[float, np.ndarray],
                k: Optional[int] = None,
                return_dist: bool = False):
        """
        Return k nearest neighbors for `y` at design `xi`.

        Parameters
        ----------
        y : array-like, shape (m, d_y) or (d_y,)
            Query outputs.
        xi : scalar or array-like
            Design key to select the per-design index.
        k : Optional[int]
            Number of neighbors; defaults to self.knn.
        return_dist : bool
            If True, also return distances and raw indices.

        Returns
        -------
        theta_neighbors : (m*k, dθ) stacked θ for all query rows
        distances, indices : optional
        """
        if not self.is_ready:
            raise RuntimeError("Call setup() before nearest().")
        key = self._key_from_xi(xi)
        if key not in self._neigh_by_xi:
            raise KeyError(f"Design {xi} not in index. Known: {list(self._neigh_by_xi.keys())}")
        y = np.atleast_2d(np.asarray(y, float))
        sc = self._scaler_by_xi[key]
        neigh = self._neigh_by_xi[key]
        k_eff = int(k or self.knn)
        d, idx = neigh.kneighbors(sc.transform(y), n_neighbors=k_eff)
        theta = self._theta_by_xi[key]
        theta_neighbors = np.vstack([theta[i] for i in idx])
        if return_dist:
            return theta_neighbors, d, idx
        return theta_neighbors

    # ---------- calibration ----------
    def calibrate(self,
                  observations,
                  resample_n: int | None = None,
                  combine: str = "stack",
                  combine_params: dict | None = None):
        """
        kNN calibration with two aggregation modes:

        combine:
          - 'stack'     : concatenate all kNN θ; optional de-duplication
          - 'intersect' : keep θ that occur at least 'min_count' times across all neighbor hits

        combine_params:
          - dedup: bool (default False) — only for 'stack'
          - theta_match_tol: float (default 1e-9) — rounding quantum for row matching/dup
          - min_count: int | None — minimum occurrences for 'intersect'
                       default: max(1, ceil(0.5 * total_blocks))   # “appear in about half of the lists”
          - use_kde: bool (default False) — if True, compute KDE log-scores and normalized weights
          - kde_bandwidth: float | None — optional bandwidth for KDE (Scott’s rule if None)

        Returns:
          dict with keys:
            'mode'    : 'knn'
            'theta'   : (n,dθ) posterior samples (resampled if requested)
            'weights' : None (stack/intersect) or KDE weights if use_kde=True
            'meta'    : dict with aggregation info (and KDE bandwidth if used)
        """
        if not self.is_ready:
            raise RuntimeError("Call setup() before calibrate().")

        combine_params = combine_params or {}
        dedup = bool(combine_params.get("dedup", False))
        tol = float(combine_params.get("theta_match_tol", 1e-9))
        use_kde = bool(combine_params.get("use_kde", True))
        kde_bw = combine_params.get("kde_bandwidth", 0.1)

        # ---------------- Collect θ-neighbors for every (y, ξ) ----------------
        theta_hits = []  # list of (n_i*k, dθ) blocks, one block per y-row (across all designs)
        for (y_obs, xi) in observations:
            key = self._key_from_xi(xi)
            if key not in self._neigh_by_xi:
                raise KeyError(f"Design {xi} not in index. Known: {list(self._neigh_by_xi.keys())}")
            yo = np.atleast_2d(np.asarray(y_obs, float))
            yo = yo[~np.isnan(yo).any(axis=1)]
            if yo.size == 0:
                continue
            sc, neigh = self._scaler_by_xi[key], self._neigh_by_xi[key]
            d, idx = neigh.kneighbors(sc.transform(yo), n_neighbors=self.knn, return_distance=True)
            # gather θ for this design
            th = self._theta_by_xi[key]
            # flatten all rows’ neighbors for this block
            theta_block = np.vstack([th[i] for i in idx])  # (n_rows*k, dθ)
            theta_hits.append(theta_block)

        if len(theta_hits) == 0:
            raise ValueError("No valid observations after NaN filtering.")

        # ---------------- Aggregation strategies ----------------
        if combine == "stack":
            theta_all = np.vstack(theta_hits)  # (sum n_i*k, dθ)

            if dedup:
                uniq, _ = self._round_rows(theta_all, tol)
                theta_out = uniq
            else:
                theta_out = theta_all

            # Optional KDE scoring on returned support
            weights = None
            meta = {"combine": "stack", "dedup": dedup, "theta_match_tol": tol}
            if use_kde and theta_out.shape[0] > 0:
                logp, w = self._kde_logweights(theta_out, bw=kde_bw)
                weights = w
                meta.update({"use_kde": True, "kde_bandwidth": kde_bw})

            # Optional resampling
            if resample_n and theta_out.shape[0] > 0:
                rng = np.random.default_rng(self.random_state)
                if weights is None:
                    take = rng.choice(theta_out.shape[0], size=int(resample_n), replace=True)
                else:
                    take = rng.choice(theta_out.shape[0], size=int(resample_n), replace=True, p=weights)
                theta_out = theta_out[take]

            self._posterior = {"mode": "knn", "theta": theta_out, "weights": weights, "meta": meta}
            return self._posterior

        elif combine == "intersect":
            # Build one big stack and count approximate matches
            big = np.vstack(theta_hits)  # (M, dθ)
            uniq, counts = self._round_rows(big, tol)

            # total neighbor lists (one per row across all designs)
            total_blocks = sum(b.shape[0] // self.knn for b in theta_hits)

            # Strictness knobs
            min_frac = float(combine_params.get("min_frac", 0.8))  # keep θ seen in ≥80% of lists
            min_count = combine_params.get("min_count", None)
            if min_count is None:
                min_count = max(1, int(np.ceil(min_frac * total_blocks)))

            # Filter by frequency
            keep = counts >= int(min_count)
            theta_out = uniq[keep]
            counts_sel = counts[keep].astype(float)

            # If nothing passed, fall back to TOP-FRACTION
            if theta_out.shape[0] == 0:
                top_frac = float(combine_params.get("top_frac", 0.1))  # keep top 10% by frequency
                k = max(1, int(np.ceil(top_frac * len(counts))))
                top_idx = np.argsort(counts)[::-1][:k]
                theta_out = uniq[top_idx]
                counts_sel = counts[top_idx].astype(float)
                meta = {"combine": "intersect", "theta_match_tol": tol,
                        "min_count": min_count, "min_frac": min_frac,
                        "fallback": f"top-{top_frac:.2f}"}
            else:
                meta = {"combine": "intersect", "theta_match_tol": tol,
                        "min_count": min_count, "min_frac": min_frac}

            # Frequency-based weights (sharpen with gamma)
            weights = None
            if theta_out.shape[0] > 0:
                gamma = float(combine_params.get("gamma", 1.0))  # 1.0=no sharpen, 2.0=stricter
                w_counts = counts_sel ** max(gamma, 1e-12)

                # Optional: KDE blending for smoother density
                if bool(combine_params.get("use_kde", False)):
                    kde_bw = combine_params.get("kde_bandwidth", 0.1)
                    _, w_kde = self._kde_logweights(theta_out, bw=kde_bw)
                    beta = float(combine_params.get("beta", 1.0))  # blend exponent for KDE
                    w = w_counts * (w_kde ** beta)
                    w = np.asarray(w, float)
                    w = w / (w.sum() if w.sum() > 0 else len(w))
                    weights = w
                    meta.update({"use_kde": True, "kde_bandwidth": kde_bw, "gamma": gamma, "beta": beta})
                else:
                    w = w_counts / (w_counts.sum() if w_counts.sum() > 0 else len(w_counts))
                    weights = w
                    meta.update({"gamma": gamma})

            # Optional resampling
            if resample_n and theta_out.shape[0] > 0:
                rng = np.random.default_rng(self.random_state)
                if weights is None:
                    take = rng.choice(theta_out.shape[0], size=int(resample_n), replace=True)
                else:
                    take = rng.choice(theta_out.shape[0], size=int(resample_n), replace=True, p=weights)
                theta_out = theta_out[take]

            self._posterior = {"mode": "knn", "theta": theta_out, "weights": weights, "meta": meta}
            return self._posterior

        else:
            raise ValueError("`combine` must be 'stack' or 'intersect'.")


    def _round_rows(self, A: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Round rows of A to multiples of `tol` and return (unique_rows, counts).
        If tol <= 0, exact matching is used.
        """
        import numpy as _np
        A = _np.asarray(A, float)
        if A.size == 0:
            return A.copy(), _np.array([], dtype=int)
        if tol <= 0:
            uniq, idx, counts = _np.unique(A, axis=0, return_index=True, return_counts=True)
            order = _np.sort(idx)
            uniq = A[order]
            counts = counts[_np.argsort(idx)]
            return uniq, counts
        R = _np.round(A / tol) * tol
        uniq, idx, counts = _np.unique(R, axis=0, return_index=True, return_counts=True)
        order = _np.sort(idx)
        uniq = R[order]
        counts = counts[_np.argsort(idx)]
        return uniq, counts

    def _kde_logweights(self, X, bw=0.5, n_max_exact=5000):
        """
        Compute KDE-based log-weights for posterior samples X.

        Args:
            X : ndarray (n, d)
                Posterior samples.
            bw : float
                Bandwidth for Gaussian kernel.
            n_max_exact : int
                Max n for exact pairwise KDE. Above this, fall back to sklearn.KernelDensity.

        Returns:
            logp : ndarray (n,)
                Log-density values at X.
            w : ndarray (n,)
                Normalized weights.
        """
        n, d = X.shape
        if n <= n_max_exact:
            # ---- Exact method (safe for small n) ----
            h2 = float(bw) ** 2
            d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)  # (n,n)
            K = np.exp(-0.5 * d2 / (h2 + 1e-18))
            sK = K.sum(axis=1) + 1e-300
            logp = np.log(sK)
            w = sK / sK.sum()
        else:
            # ---- Scalable method (sklearn KD-tree backend) ----
            kde = KernelDensity(kernel="gaussian", bandwidth=bw)
            kde.fit(X)
            logp = kde.score_samples(X)  # log density at each sample
            w = np.exp(logp - logp.max())
            w /= w.sum()

        return logp, w


    # ---------- posterior ----------
    def get_posterior(self) -> Any:
        """Return the last computed posterior dict; raises if calibrate() hasn't been called."""
        if self._posterior is None:
            raise RuntimeError("No posterior available. Run calibrate() first.")
        return self._posterior




def estimate_p_theta_knn(observed_data,
                         simulated_data,
                         xi_star,
                         knn: int = 20,
                         a_tol: float =0.05):
    """
    Estimate the posterior distribution p(θ) of θ using a k-Nearest Neighbors (kNN)
    filter on a pre-computed simulation archive, conditioned on a design ξ*.

    This method restricts the simulation archive to runs at (or near) the
    target design ξ*, then fits a kNN model in output (y) space. For each
    observed output y_obs, it retrieves the k-nearest simulated outputs and
    returns the corresponding θ values as approximate posterior samples.
    Args:
        observed_data (np.ndarray):
            Array of observed outputs y_obs (shape: n_obs × d_y).
            Must match the dimensionality of simulated outputs.
        simulated_data (list):
            A list of arrays [y, θ, ξ], containing
                - y (n × d_y): simulation output, e.g. a transformed y with only KPIs
                - θ (n × d_theta): parameters and variables to be calibrated
                - ξ (n × d_xi):  conditioning controllable factors, e.g., design,  parameters
        knn (int):
            Number of nearest neighbors to query per observed sample.
        xi_star
            Target design ξ* at which the posterior is estimated.
        a_tol (float, optional):
            Tolerance for matching simulations to ξ*. Defaults to 0.1.
            A simulation is kept if ||xi_sim - xi_star||∞ < a_tol.

    Returns:
        np.ndarray:
            θ samples from the posterior, stacked across all observed y.
            Shape: (n_obs × knn, d_theta).

    Raises:
        ValueError: If filtering leaves no simulations at ξ*.
        RuntimeError: If kNN search fails due to inconsistent dimensions.

    Notes:
        - Scaling of outputs y is performed internally via StandardScaler
          for robustness against different KPI magnitudes.
        - The parameter `knn` acts as a smoothing parameter: higher values
          broaden the posterior but reduce sharpness.
        - The choice of `a_tol` trades off strict design conditioning vs.
          sample size. Too small → few matches; too large → weaker conditioning.

    Example:
        >>> import numpy as np
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.neighbors import NearestNeighbors
        >>> # Fake simulator archive
        >>> theta_sim = np.random.uniform(-5, 5, size=(5000, 2))
        >>> xi_sim = np.zeros((5000, 1))
        >>> y_sim = np.sum(theta_sim**2, axis=1, keepdims=True) \
        ...         + 0.1*np.random.randn(5000, 1)
        >>> simulated_data = [y_sim, theta_sim, xi_sim]
        >>> # Observed data
        >>> theta_true = np.array([1.5, -2.0])
        >>> y_obs = np.sum(theta_true**2) + 0.1*np.random.randn(1)
        >>> # Estimate posterior
        >>> theta_post = estimate_p_theta_knn(
        ...     observed_data=np.array([[y_obs]]),
        ...     simulated_data=simulated_data,
        ...     knn=50,
        ...     xi_star=0.0
        ... )
        >>> theta_post.shape
        (50, 2)
        >>> theta_post.mean(axis=0)
        array([ 1.4, -2.1])  # close to true [1.5, -2.0]
    """

    # Step 1: Filter simulated datasets based on ξ = ξ*
    xi_idx = np.all(np.abs(simulated_data[2] - xi_star) < a_tol, axis=1)
    simulated_data_xi = [s[xi_idx] for s in simulated_data]

    # Step 2: fit a kNN on the (filtered) space of y. Normalize observations
    scaler = StandardScaler()
    if np.any(np.isnan(simulated_data_xi[0])):
        simulated_data_xi[0] = simulated_data_xi[0][~np.isnan(simulated_data_xi[0]).any(axis=1)]

    scaler.fit(simulated_data_xi[0])
    neigh = NearestNeighbors(n_neighbors=knn)
    neigh.fit(scaler.transform(simulated_data_xi[0]))

    # Step 3: retrieve the kNN for each observed y_i  ...... check if there are nan values in the observed datasets
    if np.any(np.isnan(observed_data)):
        observed_data = observed_data[~np.isnan(observed_data).any(axis=1)]
    dist, knn_idx = neigh.kneighbors(scaler.transform(observed_data))
    theta_set = np.vstack([simulated_data_xi[1][idx] for idx in knn_idx])
    return theta_set


























# ----------------------------
# Example usage
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # --- import your unified calibrator ---
    # from knn import KNNCalibrator
    # (Assuming KNNCalibrator is defined in the current file for this snippet.)

    def paraboloid_model(theta, xi=0.0, A=1.0, B=0.5, C=1.5):
        """Vectorized paraboloid, mild noise; supports scalar or vector xi."""
        theta = np.atleast_2d(theta).astype(float)
        x1, x2 = theta[:, 0], theta[:, 1]
        xi = np.asarray(xi, float)
        if xi.ndim == 0:
            xi = np.full(theta.shape[0], xi)
        elif xi.ndim == 2:  # if passed as (n,1)
            xi = xi.ravel()
        y = A * x1**2 + B * x1 * x2 * (1.0 + xi) + C * (x2 + xi) ** 2
        y = y + 0.2 * np.random.randn(theta.shape[0])   # small noise
        return y.reshape(-1, 1) if theta.shape[0] > 1 else np.array([y.item()])

    def theta_sampler(n, lb=-15, ub=15):
        return np.random.uniform(lb, ub, size=(n, 2))

    # --------------- Build observations (unknown process) ---------------
    N_emp = 100
    rng = np.random.default_rng(7)
    theta_target = rng.normal(3.1, 0.3, size=(N_emp, 2))   # this is unknown in practice
    experiment_designs = [-1.0, 0.0, 1.0, 3.0]

    observations = []
    for xi in experiment_designs:
        y_emp = paraboloid_model(theta=theta_target, xi=xi)  # shape (100,1) per design
        observations.append((y_emp, xi))

    # --------------- kNN calibration multiple designs ---------------
    # Use model+sampler so each design gets its own per-design kNN built on the SAME theta grid
    calib_joint = KNNCalibrator(knn=100, evaluate_model=True)
    calib_joint.setup(
        model=paraboloid_model,
        theta_sampler=theta_sampler,
        xi_list=experiment_designs,
        n_samples=100_000,
    )

    #  “intersect”
    post_joint = calib_joint.calibrate(observations=observations, combine="intersect", resample_n=5000)
    theta_post_joint = post_joint["theta"]        # (5000, 2) resampled
    # If you wanted grid + weights instead, call with resample_n=None and use post_joint["theta"], post_joint["weights"].

    # --------------- SINGLE-DESIGN calibration at xi=0.0 ---------------
    xi_star = 0.0
    y_obs_many = next(y for (y, xi) in observations if np.isclose(xi, xi_star))

    calib_single = KNNCalibrator(knn=100, evaluate_model=True)
    calib_single.setup(
        model=paraboloid_model,
        theta_sampler=lambda n: theta_sampler(n, -15, 15),
        xi_list=[xi_star],
        n_samples=100_000,
    )
    post_single = calib_single.calibrate([(y_obs_many, xi_star)], combine="stack")  # pooled kNN
    theta_post_many = post_single["theta"]  # shape: (N_emp*knn, 2)

    # --------------- Quick nearest() demo ---------------
    # Take a single observed y at xi=1.0 and fetch its 10 nearest θ:
    y_one = observations[2][0][0]  # first row at design 1.0
    theta_knn_10 = calib_joint.nearest(y_one, xi=1.0, k=10)  # (10,2) θ neighbors

    # --------------- PLOTS ---------------
    # 1) Joint posterior (resampled) vs true cloud
    plt.figure(figsize=(6, 5))
    plt.scatter(theta_post_joint[:, 0], theta_post_joint[:, 1],  s=6, alpha=0.25, label="Joint posterior (vote, resampled)")
    plt.scatter(theta_target[:, 0], theta_target[:, 1],  c="r", marker="x", s=40, label=r"θ_true samples (unknown)")
    plt.title("Unified kNN: joint combine='vote' over multiple designs")
    plt.xlabel("θ1"); plt.ylabel("θ2"); plt.legend(); plt.grid(True)
    plt.show()

    # 2) Single-design posterior (kNN pooled) vs true cloud
    plt.figure(figsize=(6, 5))
    plt.scatter(theta_post_many[:, 0], theta_post_many[:, 1], s=6, alpha=0.35, label="Single-design posterior (pooled kNN)")
    plt.scatter(theta_target[:, 0], theta_target[:, 1],  c="r", marker="x", s=40, label=r"θ_true samples (unknown)")
    plt.title(f"Unified kNN: single-design at ξ*={xi_star}")
    plt.xlabel("θ1"); plt.ylabel("θ2"); plt.legend(); plt.grid(True)
    plt.show()

    # 3) Show the 10 nearest θ for one observed y at xi=1.0 (sanity check)
    print("Ten nearest θ for one observed y at ξ=1.0:\n", theta_knn_10)
