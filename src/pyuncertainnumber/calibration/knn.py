from .calibration import *
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity


class KNNCalibrator(Calibrator):
    """Unified kNN-based calibrator for black-box models or precomputed simulations.

    Args:
        knn (int): Number of neighbors per observed row. Default: 100.
        a_tol (float): Tolerance for matching simulated :math:`\\xi` to a requested :math:`\\xi^*`
            (when reusing). A simulation is kept if :math:`\\|\\xi_{\\text{sim}} - \\xi^*\\|_\\infty < a_{\\text{tol}}`.
            Default: 0.05.
        evaluate_model (bool): If True, call the black-box model for each :math:`\\xi` in ``xi_list``
            on a shared :math:`\\theta` grid. If False, reuse ``simulated_data`` (requires y/theta/xi arrays).
            Default: False.
        random_state (int): Seed for reproducibility (affects theta_sampler and resampling). Default: 42.

    Note:
        **Setup (unified approach)**:

        - If ``evaluate_model=False`` and ``simulated_data`` is provided:

          * Reuse pre-computed simulations
          * Build a per-design kNN index by filtering rows with :math:`\\|\\xi - \\xi^*\\|_\\infty < a_{\\text{tol}}`
            for each :math:`\\xi^*` in ``xi_list``

        - If ``evaluate_model=True``:

          * Simulate :math:`y = \\text{model}(\\theta, \\xi)` for each :math:`\\xi` in ``xi_list``
          * Use a shared :math:`\\theta` grid drawn once from ``theta_sampler(n_samples)``
          * Build per-design kNN indices on this shared grid

        **Calibration workflow (single/multi-design)**:

        For each observation pair :math:`(y_{\\text{obs}}, \\xi)`:

        1. Standardize :math:`y_{\\text{obs}}` using the per-design scaler
        2. Find k nearest neighbors in y-space
        3. Map neighbor indices to :math:`\\theta` values for that design
        4. Stack :math:`\\theta` samples across all observations/designs (or apply voting/intersection)

    .. figure:: /_static/knn_illustration.png
        :alt: knn_illustration
        :align: center
        :width: 50%

        KNN calibration illustration.
    """

    def __init__(
        self, knn: int = 100, a_tol: float = 0.05, evaluate_model: bool = False
    ):

        super().__init__()
        self.knn = int(knn)
        self.a_tol = float(a_tol)
        self.evaluate_model = bool(evaluate_model)
        self.random_state = 42

        # Internal state
        self._theta_grid: Optional[np.ndarray] = (
            None  # shared grid if evaluate_model=True, else unused
        )
        self._theta_by_xi: Dict[Tuple[float, ...], np.ndarray] = (
            {}
        )  # per-design θ (may be shared ref)
        self._y_by_xi: Dict[Tuple[float, ...], np.ndarray] = {}  # per-design y
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
    def setup(
        self,
        model: Optional[
            Callable[[np.ndarray, Union[float, np.ndarray]], np.ndarray]
        ] = None,
        theta_sampler: Optional[Callable[[int], np.ndarray]] = None,
        simulated_data: Optional[Dict[str, np.ndarray]] = None,
        xi_list: Optional[List[Union[float, np.ndarray]]] = None,
        n_samples: int = 10000,
    ):
        """Prepare per-design kNN structures by either reusing simulated_data or simulating for each design.

        Args:
            model (callable, optional): Black-box simulator with signature ``model(theta, xi) -> y``
                (vectorized over theta).
            theta_sampler (callable, optional): Sampler for :math:`\\theta`; required when ``evaluate_model=True``.
            simulated_data (dict, optional): Dict with keys {"y": (n, dy), "theta": (n, dθ), "xi": (n, dξ)}
                when reusing sims.
            xi_list (list, optional): List of designs; each item can be scalar or array-like.
                If None, defaults to [0.0].
            n_samples (int): Number of :math:`\\theta` samples to draw when ``evaluate_model=True``. Default: 10000.
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
                raise ValueError(
                    "evaluate_model=False requires `simulated_data` with keys 'y','theta','xi'."
                )

            self._sim_y = np.asarray(simulated_data["y"], float)
            self._sim_theta = np.asarray(simulated_data["theta"], float)
            self._sim_xi = np.asarray(simulated_data.get("xi", None), float)
            if self._sim_xi is None:
                raise ValueError(
                    "`simulated_data` must include 'xi' to filter per design."
                )

            for xi in xi_list:
                key = self._key_from_xi(xi)
                mask = np.all(
                    np.abs(self._sim_xi - np.atleast_1d(xi)) < self.a_tol, axis=1
                )
                y_xi = self._sim_y[mask]
                theta_xi = self._sim_theta[mask]
                if y_xi.size == 0:
                    raise ValueError(
                        f"No simulations matched design {xi} within tolerance a_tol={self.a_tol}."
                    )
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
                raise ValueError(
                    "evaluate_model=True requires `model` and `theta_sampler`."
                )
            self._theta_grid = np.asarray(theta_sampler(int(n_samples)), float)
            if self._theta_grid.ndim != 2:
                raise ValueError(
                    "theta_sampler must return a 2D array (n_samples, dθ)."
                )

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
                self._grid_idx_by_xi[key] = (
                    grid_idx  # maps local row j -> global grid index grid_idx[j]
                )

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
    def nearest(
        self,
        y: Union[np.ndarray, List[float]],
        xi: Union[float, np.ndarray],
        k: Optional[int] = None,
        return_dist: bool = False,
    ):
        """Return k nearest neighbors for y at design xi.

        Args:
            y (array-like): Query outputs, shape (m, d_y) or (d_y,).
            xi (scalar or array-like): Design key to select the per-design index.
            k (int, optional): Number of neighbors; defaults to ``self.knn``.
            return_dist (bool): If True, also return distances and raw indices. Default: False.

        Returns:
            theta_neighbors (ndarray): Shape (m*k, dθ) stacked :math:`\\theta` for all query rows.
            distances (ndarray, optional): Returned if ``return_dist=True``.
            indices (ndarray, optional): Returned if ``return_dist=True``.
        """
        if not self.is_ready:
            raise RuntimeError("Call setup() before nearest().")
        key = self._key_from_xi(xi)
        if key not in self._neigh_by_xi:
            raise KeyError(
                f"Design {xi} not in index. Known: {list(self._neigh_by_xi.keys())}"
            )
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
    def calibrate(
        self,
        observations,
        resample_n: int | None = None,
        combine: str = "stack",
        combine_params: dict | None = None,
    ):
        """Run kNN calibration and aggregate posterior θ across neighbor-hit blocks.

        Args:
            observations: Observed simulator or model outputs to calibrate against.

            resample_n (int | None): If set, resample posterior θ samples to this size.
                If `None`, return all aggregated θ without resampling.

            combine (str): Aggregation mode. One of:

                - **'stack'**: concatenate all kNN θ; optional de-duplication.

                - **'intersect'**: retain θ hit at least `min_count` times across neighbor blocks.

            combine_params (dict | None): Optional parameters controlling aggregation and KDE weighting.

                Supported keys:

                - **dedup** (bool): Default `False`. Remove duplicate θ (only for 'stack').

                - **theta_match_tol** (float): Default `1e-9`. Tolerance or rounding quantum for comparing/merging θ values.

                - **min_count** (int | None): Minimum occurrences for 'intersect'. Default is `max(1, ceil(0.5 * total_blocks))`, meaning θ must appear in about half of neighbor lists.

                - **use_kde** (bool): Default `False`. If `True`, fit KDE on aggregated θ to compute log-scores and normalized weights.

                - **kde_bandwidth** (float | None): Bandwidth for KDE. If `None` (default), use Scott's rule.

        tip:
            Two aggregation modes are supported:

            - **stack**: Concatenate all kNN θ into a single array. Supports optional de-duplication of nearly identical θ values.

            - **intersect**: Keep θ values that occur in at least `min_count` neighbor blocks across all observations/design points (default ≈ half of all blocks).

            Optional density weighting via KDE can be applied after aggregation to compute normalized posterior weights.


        Returns:
            dict: A dictionary with keys:
                - **'mode'** (str): Always `'knn'`.
                - **'theta'** (ndarray): Posterior samples of shape `(N, dθ)`;
                  resampled if `resample_n` is provided.
                - **'weights'** (ndarray | None): `None` for stack/intersect,
                  or a length-`N` array of KDE weights if `use_kde=True`.
                - **'meta'** (dict): Aggregation info; may include KDE bandwidth
                  if density weighting is used.
        """

        if not self.is_ready:
            raise RuntimeError("Call setup() before calibrate().")

        combine_params = combine_params or {}
        dedup = bool(combine_params.get("dedup", False))
        tol = float(combine_params.get("theta_match_tol", 1e-9))
        use_kde = bool(combine_params.get("use_kde", True))
        kde_bw = combine_params.get("kde_bandwidth", 0.1)

        # ---------------- Collect θ-neighbors for every (y, ξ) ----------------
        theta_hits = (
            []
        )  # list of (n_i*k, dθ) blocks, one block per y-row (across all designs)
        for y_obs, xi in observations:
            key = self._key_from_xi(xi)
            if key not in self._neigh_by_xi:
                raise KeyError(
                    f"Design {xi} not in index. Known: {list(self._neigh_by_xi.keys())}"
                )
            yo = np.atleast_2d(np.asarray(y_obs, float))
            yo = yo[~np.isnan(yo).any(axis=1)]
            if yo.size == 0:
                continue
            sc, neigh = self._scaler_by_xi[key], self._neigh_by_xi[key]
            d, idx = neigh.kneighbors(
                sc.transform(yo), n_neighbors=self.knn, return_distance=True
            )
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
                    take = rng.choice(
                        theta_out.shape[0], size=int(resample_n), replace=True
                    )
                else:
                    take = rng.choice(
                        theta_out.shape[0],
                        size=int(resample_n),
                        replace=True,
                        p=weights,
                    )
                theta_out = theta_out[take]

            self._posterior = {
                "mode": "knn",
                "theta": theta_out,
                "weights": weights,
                "meta": meta,
            }
            return self._posterior

        elif combine == "intersect":
            # Build one big stack and count approximate matches
            big = np.vstack(theta_hits)  # (M, dθ)
            uniq, counts = self._round_rows(big, tol)

            # total neighbor lists (one per row across all designs)
            total_blocks = sum(b.shape[0] // self.knn for b in theta_hits)

            # Strictness knobs
            min_frac = float(
                combine_params.get("min_frac", 0.8)
            )  # keep θ seen in ≥80% of lists
            min_count = combine_params.get("min_count", None)
            if min_count is None:
                min_count = max(1, int(np.ceil(min_frac * total_blocks)))

            # Filter by frequency
            keep = counts >= int(min_count)
            theta_out = uniq[keep]
            counts_sel = counts[keep].astype(float)

            # If nothing passed, fall back to TOP-FRACTION
            if theta_out.shape[0] == 0:
                top_frac = float(
                    combine_params.get("top_frac", 0.1)
                )  # keep top 10% by frequency
                k = max(1, int(np.ceil(top_frac * len(counts))))
                top_idx = np.argsort(counts)[::-1][:k]
                theta_out = uniq[top_idx]
                counts_sel = counts[top_idx].astype(float)
                meta = {
                    "combine": "intersect",
                    "theta_match_tol": tol,
                    "min_count": min_count,
                    "min_frac": min_frac,
                    "fallback": f"top-{top_frac:.2f}",
                }
            else:
                meta = {
                    "combine": "intersect",
                    "theta_match_tol": tol,
                    "min_count": min_count,
                    "min_frac": min_frac,
                }

            # Frequency-based weights (sharpen with gamma)
            weights = None
            if theta_out.shape[0] > 0:
                gamma = float(
                    combine_params.get("gamma", 1.0)
                )  # 1.0=no sharpen, 2.0=stricter
                w_counts = counts_sel ** max(gamma, 1e-12)

                # Optional: KDE blending for smoother density
                if bool(combine_params.get("use_kde", False)):
                    kde_bw = combine_params.get("kde_bandwidth", 0.1)
                    _, w_kde = self._kde_logweights(theta_out, bw=kde_bw)
                    beta = float(
                        combine_params.get("beta", 1.0)
                    )  # blend exponent for KDE
                    w = w_counts * (w_kde**beta)
                    w = np.asarray(w, float)
                    w = w / (w.sum() if w.sum() > 0 else len(w))
                    weights = w
                    meta.update(
                        {
                            "use_kde": True,
                            "kde_bandwidth": kde_bw,
                            "gamma": gamma,
                            "beta": beta,
                        }
                    )
                else:
                    w = w_counts / (
                        w_counts.sum() if w_counts.sum() > 0 else len(w_counts)
                    )
                    weights = w
                    meta.update({"gamma": gamma})

            # Optional resampling
            if resample_n and theta_out.shape[0] > 0:
                rng = np.random.default_rng(self.random_state)
                if weights is None:
                    take = rng.choice(
                        theta_out.shape[0], size=int(resample_n), replace=True
                    )
                else:
                    take = rng.choice(
                        theta_out.shape[0],
                        size=int(resample_n),
                        replace=True,
                        p=weights,
                    )
                theta_out = theta_out[take]

            self._posterior = {
                "mode": "knn",
                "theta": theta_out,
                "weights": weights,
                "meta": meta,
            }
            return self._posterior

        else:
            raise ValueError("`combine` must be 'stack' or 'intersect'.")

    def _round_rows(self, A: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
        """Round rows of A to multiples of tol and return (unique_rows, counts).

        Args:
            A (ndarray): Input array to process.
            tol (float): Tolerance for rounding. If tol <= 0, exact matching is used.

        Returns:
            tuple: (unique_rows, counts) where unique_rows are the deduplicated rows
                and counts are the occurrence counts.
        """
        import numpy as _np

        A = _np.asarray(A, float)
        if A.size == 0:
            return A.copy(), _np.array([], dtype=int)
        if tol <= 0:
            uniq, idx, counts = _np.unique(
                A, axis=0, return_index=True, return_counts=True
            )
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
        """Compute KDE-based log-weights for posterior samples X.

        Args:
            X (ndarray): Posterior samples, shape (n, d).
            bw (float): Bandwidth for Gaussian kernel. Default: 0.5.
            n_max_exact (int): Max n for exact pairwise KDE. Above this, fall back to
                sklearn.KernelDensity. Default: 5000.

        Returns:
            tuple:
                - **logp** (ndarray): Log-density values at X, shape (n,).
                - **w** (ndarray): Normalized weights, shape (n,).
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


def estimate_p_theta_knn(
    observed_data, simulated_data, xi_star, knn: int = 20, a_tol: float = 0.05
):
    """Estimate the posterior distribution p(θ) using k-Nearest Neighbors (kNN) on a simulation archive.

    This method restricts the simulation archive to runs at (or near) the target design :math:`\\xi^*`, 
    then fits a kNN model in output (y) space. For each observed output y_obs, it retrieves the 
    k-nearest simulated outputs and returns the corresponding :math:`\\theta` values as approximate 
    posterior samples.

    Args:
        observed_data (ndarray): Array of observed outputs y_obs, shape (n_obs, d_y).
            Must match the dimensionality of simulated outputs.
        simulated_data (list): List of arrays [y, θ, ξ], containing:
            
            - **y** (ndarray): Simulation output, shape (n, d_y), e.g., transformed y with only KPIs
            - **θ** (ndarray): Parameters and variables to be calibrated, shape (n, d_theta)
            - **ξ** (ndarray): Conditioning controllable factors, shape (n, d_xi), e.g., design parameters
            
        knn (int): Number of nearest neighbors to query per observed sample. Default: 20.
        xi_star (scalar or array-like): Target design :math:`\\xi^*` at which the posterior is estimated.
        a_tol (float): Tolerance for matching simulations to :math:`\\xi^*`. Default: 0.05.
            A simulation is kept if :math:`\\|\\xi_{\\text{sim}} - \\xi^*\\|_\\infty < a_{\\text{tol}}`.

    Returns:
        ndarray: :math:`\\theta` samples from the posterior, stacked across all observed y.
            Shape: (n_obs × knn, d_theta).

    Raises:
        ValueError: If filtering leaves no simulations at :math:`\\xi^*`.
        RuntimeError: If kNN search fails due to inconsistent dimensions.

    Note:
        - Scaling of outputs y is performed internally via StandardScaler for robustness 
          against different KPI magnitudes.
        - The parameter ``knn`` acts as a smoothing parameter: higher values broaden the 
          posterior but reduce sharpness.
        - The choice of ``a_tol`` trades off strict design conditioning vs. sample size. 
          Too small → few matches; too large → weaker conditioning.

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
        simulated_data_xi[0] = simulated_data_xi[0][
            ~np.isnan(simulated_data_xi[0]).any(axis=1)
        ]

    scaler.fit(simulated_data_xi[0])
    neigh = NearestNeighbors(n_neighbors=knn)
    neigh.fit(scaler.transform(simulated_data_xi[0]))

    # Step 3: retrieve the kNN for each observed y_i  ...... check if there are nan values in the observed datasets
    if np.any(np.isnan(observed_data)):
        observed_data = observed_data[~np.isnan(observed_data).any(axis=1)]
    dist, knn_idx = neigh.kneighbors(scaler.transform(observed_data))
    theta_set = np.vstack([simulated_data_xi[1][idx] for idx in knn_idx])
    return theta_set
