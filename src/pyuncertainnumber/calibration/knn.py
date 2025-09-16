
from calibration import *



class KNNCalibrator(Calibrator):
    r""" General-purpose calibration for black-box / given simulation data via k-Nearest Neighbors.
    .. math::
        p(\theta \mid y_{obs}, \xi^*) \;\propto\; \sum_{i=1}^N K_h \!\left( y_{obs} - g(\theta_i, \xi^*) \right)

    where :math:`g(\theta, \xi)` is a black-box simulator producing outputs :math:`y`,
    and :math:`K_h(\cdot)` is either the kNN indicator kernel (single-design) or
    a Gaussian kernel (multi-design).

    Modes
    -----
    - **Single-design (kNN)**:
      calibration is performed by nearest-neighbor search in y-space at a single design :math:`\xi^*`.
    - **Multi-design (joint kernel)**:
      assumes the same unknown :math:`\theta` generated observations across multiple designs,
      and fuses evidence via product of Gaussian kernels.

    Args
    ----
    knn : int
        Number of neighbors for kNN (single-design).
    a_tol : float
        Tolerance when matching simulated designs to the target design.
    kernel_bandwidth : float, optional
        Kernel bandwidth in y-space (multi-design). If None, auto-estimated.

    Setup Parameters
    ----------------
    Either provide:

    - ``simulated_data`` : dict with keys
        - ``"y"`` : array of simulated outputs (n, d_y)
        - ``"theta"`` : array of input parameters (n, d_theta)
        - ``"xi"`` : array of design settings (n, d_xi), optional
    - or (``model + theta_sampler`` [+ ``xi_sampler``]) to generate simulations on-the-fly.

    The list of designs ``xi_list`` decides the mode:
      - ``len(xi_list) == 1`` → single-design (kNN)
      - ``len(xi_list) > 1``  → multi-design (joint kernel)

    Returns
    -------
    - Single-design: posterior θ samples (array)
    - Multi-design: (θ_grid, weights) if resample_n=None, or (θ_resampled, weights) if resample_n given

    Tip
    ---
    Use kNN + pre-existing input-output simulations for very fast calibration;
    Consider joint kernel mode for Bayesian model updating across multiple experimental designs.

    Caution
    -------
    The joint kernel assumes all observations arise from the same θ.
    If you want to handle multiple θ sources, run calibration separately
    for each source and overlay the results.

    Example
    -------
    >>> def paraboloid_model(theta, xi=0.0):
    ...     x1, x2 = theta
    ...     return np.array([x1**2 + 0.5*x1*x2 + (x2+xi)**2])
    >>> def theta_sampler(n):
    >>>     return np.random.uniform(-5, 5, size=(n, 2))
    >>> calib = KNNCalibrator(knn=100)
    >>> xi_base = 1.0
    >>> calib.setup(model=paraboloid_model, theta_sampler=theta_sampler, xi_list=[0.0], n_samples=5000)
    >>> true_thetas = [[-4, 2.5],[-4.1, 2.4],[-4.2, 2.2],[-3.4, 2.2]]
    >>> y_obs = paraboloid_model(true_thetas, xi=xi_base).reshape(1, -1)
    >>> theta_post = calib.calibrate(observations=[(y_obs, xi_base)])
    """

    def __init__(self, knn: int = 100, a_tol: float = 1e-3, kernel_bandwidth: Optional[float] = None):
        super().__init__()
        self.knn = knn
        self.a_tol = a_tol
        self.kernel_bandwidth = kernel_bandwidth

        # internal state
        self._mode = None
        self._scaler = None
        self._neigh = None
        self._theta_sim_single = None
        self._theta_sim_joint = None
        self._y_sim_by_xi = {}
        self._dy_joint = None
        self._posterior = None

    # ---------- utilities ----------
    @staticmethod
    def _key_from_xi(xi) -> Tuple[float, ...]:
        return tuple(np.atleast_1d(np.asarray(xi, float)).ravel())


    # ---------- setup ----------
    def setup(self,
              model: Optional[Callable] = None,
              theta_sampler: Optional[Callable[[int], np.ndarray]] = None,
              xi_sampler: Optional[Callable[[int], np.ndarray]] = None,
              simulated_data: Optional[Dict[str, np.ndarray]] = None,
              xi_list: Optional[List] = None,
              n_samples: int = 10000,
              ):

        """setting up the calibration model"""
        if xi_list is None: # if no design is provided --> default
            xi_list = [0.0]

        if simulated_data is not None:  # if simulation input-output data is provided ---- > acquire simulations
            y_sim = simulated_data["y"]
            theta_sim = simulated_data["theta"]
            xi_sim = simulated_data.get("xi", np.zeros((len(theta_sim), 1)))
        else:
            if model is None or theta_sampler is None: # if we have no data and no model or no sampler error
                raise ValueError("Provide either simulated_data or (model + theta_sampler).")
            theta_sim = theta_sampler(n_samples)  # sample from prior, e.g., uniformly inputs (params & vars)
            xi_sim = xi_sampler(n_samples) if xi_sampler else np.zeros((n_samples, 1))  # sample design
            y_sim = np.vstack([np.atleast_1d(model(th, xi)).astype(float)
                               for th, xi in zip(theta_sim, xi_sim)]) # get model response


        if len(xi_list) == 1: # single-design
            self._mode = "single"
            xi_star = np.asarray(xi_list[0])
            mask = np.all(np.abs(xi_sim - xi_star) < self.a_tol, axis=1)
            y_sim, theta_sim = y_sim[mask], theta_sim[mask]
            mask = ~np.isnan(y_sim).any(axis=1)
            y_sim, theta_sim = y_sim[mask], theta_sim[mask]

            self._theta_sim_single = theta_sim
            self._scaler = StandardScaler().fit(y_sim)
            self._neigh = NearestNeighbors(n_neighbors=self.knn).fit(self._scaler.transform(y_sim))


        else: # multi-design
            self._mode = "joint"
            self._theta_sim_joint = np.asarray(theta_sim, float)
            self._y_sim_by_xi.clear()
            for xi in xi_list:
                key = self._key_from_xi(xi)
                y_sim_xi = np.vstack([np.atleast_1d(model(th, xi)).astype(float)  for th in theta_sim]) if model else y_sim
                self._dy_joint = y_sim_xi.shape[1]
                self._y_sim_by_xi[key] = y_sim_xi

            if self.kernel_bandwidth is None:
                all_y = np.vstack(list(self._y_sim_by_xi.values()))
                sigma = np.std(all_y, axis=0).mean() + 1e-12
                n = all_y.shape[0]
                self.kernel_bandwidth = sigma * n ** (-1.0 / (4 + self._dy_joint))

        self.is_ready = True

    # ---------- calibration ----------
    def calibrate(self, observations: List[Tuple[np.ndarray, np.ndarray]], resample_n: Optional[int] = None):
        """calibration method """

        if not self.is_ready:
            raise RuntimeError("Call setup() before calibrate().")

        if self._mode == "single":  # if single data set
            y_obs = np.atleast_2d(np.asarray(observations[0][0], float))
            y_obs = y_obs[~np.isnan(y_obs).any(axis=1)]
            _, idx = self._neigh.kneighbors(self._scaler.transform(y_obs))
            self._posterior = np.vstack([self._theta_sim_single[i] for i in idx])
            return self._posterior

        elif self._mode == "joint":
            h2 = (self.kernel_bandwidth ** 2) + 1e-18
            logw = np.zeros(self._theta_sim_joint.shape[0])
            for y_obs, xi in observations:
                xi_key = self._key_from_xi(xi)
                if xi_key not in self._y_sim_by_xi:
                    raise KeyError("Design {xi} not in joint model. Known: {list(self._y_sim_by_xi.keys())}")
                y_sim = self._y_sim_by_xi[xi_key]
                for y in np.atleast_2d(y_obs):
                    r2 = np.sum((y_sim - y) ** 2, axis=1)
                    logw += -0.5 * r2 / h2
            w = np.exp(logw - np.max(logw))
            w /= np.sum(w) if np.sum(w) > 0 else len(w)
            if resample_n is None:
                self._posterior = (self._theta_sim_joint, w)
            else:
                idx = np.random.choice(len(w), size=resample_n, replace=True, p=w)
                self._posterior = (self._theta_sim_joint[idx], w)
            return self._posterior

        return None

    # ---------- posterior ----------
    def get_posterior(self) -> Any:
        return self._posterior