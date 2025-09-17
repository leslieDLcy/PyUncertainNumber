from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import numpy as np
import inspect


class BayesOpt_args_signature:
    """Bayesian Optimisation class with original function signature (arguments signature).

    It requires the ``design_bound`` to be a dictionary, e.g. {'x1': (0, 1), 'x2': (0, 1)}.

    args:
        f (callable): the target function to be optimised, should have individual function signature. See notes.

        design_bounds (dict): the bounds for the design space, e.g. {'x1': (0, 1), 'x2': (0, 1)}

        task (str): either 'minimisation' or 'maximisation'

        acquisition_function (str or callable, optional): the acquisition function to be used, e.g. 'UCB', 'EI', 'PI'. If None, defaults to 'UCB'.

        num_explorations (int, optional): the number of initial exploration points. Defaults to 100.

        num_iterations (int, optional): the number of iterations to run the optimisation. Defaults to 100.

    note:
        Acquisition functions can be either a string (e.g. 'UCB', 'EI', 'PI') or a callable function.
        'UCB' stands for Upper Confidence Bound, 'EI' for Expected Improvement, and 'PI' for Probability of Improvement.
        If a string is provided, the parameter for the acquisition function can be passed as an additional
        argument to the class constructor. For example, for 'UCB', you can pass a `kappa` value, and for 'EI' or 'PI', you can pass an `xi` value.
        For low-level controls, if a callable function is provided, it should already be parameterised.

        About the function signature of $f$, by default it should be expecting individual arguments in the form of $f(x_0, x_1, \ldots, x_n)$, often
        one needs to write a wrapper function to unpack the input arguments when working with a black-box model, which typically has vectorisation calling signature.
        Also, one can specify the `xc_bound` accordingly. When `EpistemicDomain` is used as a shortcut to specify the `xc_bound`, the keys will be automatically
        mapped to the corresponding arguments of the function.

    example:
        >>> import numpy as np
        >>> from pyuncertainnumber.opt.bo import BayesOpt_args_signature
        >>> def black_box_function(x):
        ...     return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2 / 10) + 1 / (x**2 + 1)
        >>> bo = BayesOpt_args_signature(
        ...     f=black_box_function,
        ...     dimension=1,
        ...     design_bounds={'x': (-2, 10)},
        ...     task='maximisation',
        ...     num_explorations=3,
        ...     num_iterations=20
        ... )
        >>> bo.run(verbose=True)
        >>> print(bo.optimal)  # get the optimal parameters and target value

    .. admonition:: Implementation

        This represents the original function signature. The range of the design space, defined in `design_bound`, is a dictionary mapping parameter names to their bounds.
        However, in later versions, such as the `BayesOpt` class, the design bounds can be specified as a list or 2D numpy array of shape (n, 2), and the function signature can be automatically detected.

        example:
        >>> ed = EpistemicDomain(pba.I(-5, 5), pba.I(-5, 5))
        >>> BayesOpt(f=foo,
        ...     design_bounds= ed.to_BayesOptBounds(func_signature="arguments"),  # the trick
        ...     task='maximisation',
        ...     num_explorations=3,
        ...     num_iterations=20,
        ... )



    """

    # TODO add a descriptor for `task`
    def __init__(
        self,
        f,
        design_bounds: dict,
        task,
        acquisition_function="UCB",
        num_explorations=100,
        num_iterations=100,
    ):

        self.task = task  # either minimisation or maximisation
        self.num_explorations = num_explorations  # initial exploration points
        self.num_iterations = num_iterations
        self.design_bounds = design_bounds  # the bounds for the design space
        self.acquisition_function = self.parse_acq(acquisition_function)
        self.f = f  # the function to be optimised

    #     self.transform_xc_bounds()

    # def transform_xc_bounds(
    #     self,
    # ):
    #     if "0" in self.design_bounds.keys():
    #         self.design_bounds = rekey_bounds_by_func(self.design_bounds, self.f)

    def parse_acq(self, acq, parameter=None):
        """parse the acquisition function

        args:
            acq (str or callable): the acquisition function to be used. See notes above.
            parameter (float, optional): parameter for the acquisition function, e.g. kappa for UCB, xi for EI and PI
        """
        if isinstance(acq, str):
            if acq == "UCB":
                kappa = parameter if parameter else 10.0
                return acquisition.UpperConfidenceBound(kappa=kappa)
            elif acq == "EI":
                xi = parameter if parameter else 0.01
                return acquisition.ExpectedImprovement(xi=xi)
            elif acq == "PI":
                xi = parameter if parameter else 0.01
                return acquisition.ProbabilityOfImprovement(xi=xi)
            else:
                raise ValueError(f"Unknown acquisition function: {acq}")
        else:
            return acq

    @property
    def f(self):
        """return the function to be optimised"""
        return self._f

    @f.setter
    def f(self, f):
        if self.task == "maximisation":
            self._f = f
        elif self.task == "minimisation":
            from functools import wraps

            if self.task == "maximisation":
                self._f = f
            elif self.task == "minimisation":

                @wraps(f)  # <-- preserves f's signature for inspect.signature
                def _f(*args, **kwargs):
                    return -f(*args, **kwargs)

                self._f = _f

    def get_results(self):
        """inspect the results, to save or not"""

        self._optimal_dict = {}
        # TODO serialise the dict, plus the undering GP model

        if self.task == "maximisation":
            bo_all_dict = {
                "Xc_params": self.optimizer.space.params.tolist(),
                "target_array": self.optimizer.space.target.tolist(),
                "optimal_Xc": list(self.optimizer.max["params"].values()),
                "optimal_target": self.optimizer.max["target"],
            }
        elif self.task == "minimisation":  # the second flip
            target_arr = self.optimizer.space.target.copy()
            target_arr[:] *= -1

            optimal_index = np.argmin(target_arr)
            optimal_target = np.min(target_arr)
            optimal_Xc = self.optimizer.space.params[optimal_index]

            bo_all_dict = {
                "Xc_params": self.optimizer.space.params.tolist(),
                "target_array": target_arr.tolist(),
                "optimal_Xc": optimal_Xc.tolist(),
                "optimal_target": optimal_target.tolist(),
            }

        self._optimal_dict["xc"] = bo_all_dict["optimal_Xc"]
        self._optimal_dict["target"] = bo_all_dict["optimal_target"]
        self._all_results = bo_all_dict

    def run(self, **kwargs):
        """run the Bayesian optimisation process.

        args:
            verbose (bool, optional): whether to print the progress. Defaults to False. Use 'verbose=True' to see the progress.

            **kwargs: additional low--level arguments to be passed to the BayesianOptimization constructor.

        example:
            >>> foo.run(verbose=True)
        """

        self.optimizer = BayesianOptimization(
            f=self.f,
            pbounds=self.design_bounds,
            acquisition_function=self.acquisition_function,
            random_state=42,
            allow_duplicate_points=True,
            **kwargs,
        )

        try:
            # initial exploration of the design space
            self.optimizer.maximize(
                init_points=self.num_explorations,
                n_iter=0,
            )
        except:
            pass

        # * _________________ run the BO iterations to get the optimal Xc
        for _ in range(self.num_iterations):
            next_point = self.optimizer.suggest()
            target = self._f(**next_point)
            self.optimizer.register(params=next_point, target=target)
            # print(target, next_point)

        # * _________________ compile the results
        self.get_results()

    @property
    def optimal(self) -> dict:
        """return the optimal parameters and target value as a dictionary"""
        return self._optimal_dict

    @property
    def optimal_xc(self) -> np.ndarray:
        """return the optimal design points (xc)"""
        return np.squeeze(self._optimal_dict["xc"])

    @property
    def optimal_target(self) -> np.ndarray:
        """return the optimal target value f(xc*)"""
        return self._optimal_dict["target"]


class BayesOpt_iterable_signature(BayesOpt_args_signature):
    """Bayesian Optimisation class for iterable function style

    See `BayesOpt` for additional details.
    """

    def __init__(
        self,
        f,
        design_bounds: list[list[float]],
        dimension,
        task,
        acquisition_function="UCB",
        num_explorations=100,
        num_iterations=100,
    ):

        wrapped, pbounds, _ = as_bayes_opt(f, design_bounds, prefix="x")

        super().__init__(
            wrapped,
            pbounds,
            dimension,
            task,
            acquisition_function,
            num_explorations,
            num_iterations,
        )


class BayesOpt_vectorised_signature(BayesOpt_args_signature):
    """Bayesian Optimisation class for vectorised function style

    See `BayesOpt` for additional details.
    """

    def __init__(
        self,
        f,
        design_bounds: list[list[float]],
        task,
        acquisition_function="UCB",
        num_explorations=100,
        num_iterations=100,
    ):

        wrapped, pbounds, _ = as_bayes_opt_2d(f, design_bounds, prefix="x")

        super().__init__(
            wrapped,
            pbounds,
            task,
            acquisition_function,
            num_explorations,
            num_iterations,
        )


class BayesOpt(BayesOpt_args_signature):
    """Bayesian Optimisation class with automatic function signature detection

    The go to class for Bayesian Optimisation

    args:
        f (callable): the target function to be optimised, it could be vectorised or iterable signature but NOT arguments signature. See notes.

        design_bounds (list | np.ndarray): the bounds for the design space, e.g. [[0, 1], [0, 1]]

        task (str): either 'minimisation' or 'maximisation'

        acquisition_function (str or callable, optional): the acquisition function to be used, e.g. 'UCB', 'EI', 'PI'. If None, defaults to 'UCB'.

        num_explorations (int, optional): the number of initial exploration points. Defaults to 100.

        num_iterations (int, optional): the number of iterations to run the optimisation. Defaults to 100.

    note:
        Acquisition functions can be either a string (e.g. 'UCB', 'EI', 'PI') or a callable function.
        'UCB' stands for Upper Confidence Bound, 'EI' for Expected Improvement, and 'PI' for Probability of Improvement.
        If a string is provided, the parameter for the acquisition function can be passed as an additional
        argument to the class constructor. For example, for 'UCB', you can pass a `kappa` value, and for 'EI' or 'PI', you can pass an `xi` value.
        For low-level controls, if a callable function is provided, it should already be parameterised.

        About the function signature of $f$, by default it should be expecting individual arguments in the form of $f(x_0, x_1, \ldots, x_n)$, often
        one needs to write a wrapper function to unpack the input arguments when working with a black-box model, which typically has vectorisation calling signature.
        Also, one can specify the `xc_bound` accordingly. When `EpistemicDomain` is used as a shortcut to specify the `xc_bound`, the keys will be automatically
        mapped to the corresponding arguments of the function.

        The function is assumed to be a vectorised or iterable signature, indicating a f(x) with only one argument.

    example:
        >>> import numpy as np
        >>> from pyuncertainnumber.opt.bo import BayesOpt
        >>> def foo_vec(x):
        ...     return x[:, 0] ** 3 + x[:, 1] + x[:, 2]
        >>> bo = BayesOpt(
        ...     f=foo_vec,
        ...     design_bounds=[(-2, 2), (-3, 3), (-1, 1)],
        ...     task='maximisation',
        ...     num_explorations=3,
        ...     num_iterations=20
        ... )
        >>> bo.run(verbose=True)
        >>> print(bo.optimal)  # get the optimal parameters and target value

    .. admonition:: Implementation

        The range of the design space is defined by `design_bounds`, which is a 2D numpy array with shape (n, 2), where n is the number of parameters.
        For consistency, it is recommended to use the class `EpistemicDomain.to_BayesOptBounds()` to automatically take care of the format of the bounds.

        example:
        >>> BayesOpt(f=foo_vec,
        ...     design_bounds= [(-2, 2), (-3, 3), (-1, 1)],
        ...     task='maximisation',
        ...     num_explorations=3,
        ...     num_iterations=20,
        ... )
    """

    def __init__(
        self,
        f,
        design_bounds: list | np.ndarray,
        task,
        acquisition_function="UCB",
        num_explorations=100,
        num_iterations=100,
    ):
        wrapped, pbounds, _ = as_bayes_opt_auto(f, design_bounds)

        super().__init__(
            wrapped,
            pbounds,
            task,
            acquisition_function,
            num_explorations,
            num_iterations,
        )


def check_argument_count(func):
    # Get the function signature
    sig = inspect.signature(func)
    # Count the number of non-default parameters
    param_count = len(
        [p for p in sig.parameters.values() if p.default == inspect.Parameter.empty]
    )
    return "Single argument" if param_count == 1 else "Multiple arguments"


def transform_func(fb, **kwargs):
    args = [p for p in kwargs.values()]
    return fb(args)


import inspect
from collections import OrderedDict


def rekey_bounds_by_func(bounds_indexed: dict, func, *, allow_extras=False):
    """
    Convert bounds like {'0': (lo,hi), '1': (lo,hi), ...}
    into {'param_name': (lo,hi), ...} using func's signature.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Supported kinds: positional-only, positional-or-keyword, keyword-only.
    # (We disallow *args/**kwargs because they cannot be bounded cleanly.)
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        raise ValueError("Functions with *args are not supported for bounds mapping.")
    if any(p.kind == p.VAR_KEYWORD for p in params):
        raise ValueError(
            "Functions with **kwargs are not supported for bounds mapping."
        )

    # Parameter order is the function's declared order
    param_names = [
        p.name
        for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]

    # Sort incoming bounds by their numeric key: "0","1","2",...
    try:
        sorted_bounds = sorted(bounds_indexed.items(), key=lambda kv: int(kv[0]))
    except ValueError:
        raise ValueError("All bounds keys must be numeric strings like '0','1',...")

    if len(sorted_bounds) < len(param_names):
        raise ValueError(
            f"Not enough bounds: got {len(sorted_bounds)} for {len(param_names)} parameters {param_names}"
        )
    if not allow_extras and len(sorted_bounds) > len(param_names):
        raise ValueError(
            f"Too many bounds: got {len(sorted_bounds)} for {len(param_names)} parameters {param_names} "
            "(pass allow_extras=True to ignore extras)."
        )

    # Map 1:1 in order; if extras exist and allowed, ignore the tail
    sorted_bounds = sorted_bounds[: len(param_names)]

    # Build renamed OrderedDict to preserve function param order
    renamed = OrderedDict(
        (param_names[i], sorted_bounds[i][1]) for i in range(len(param_names))
    )
    return renamed


def as_bayes_opt(f_vec, bounds_list, prefix="x"):
    """
    Wrap a vector/array objective f_vec(x) so it can be used by bayes_opt,
    which calls f(**kwargs) with named scalar params.

    Parameters
    ----------
    f_vec : callable
        Objective expecting a 1-D array-like x of length d.
        May also be vectorized and return a numpy array; we coerce to float.
    bounds_list : list[tuple[float, float]]
        [(low_0, high_0), ..., (low_{d-1}, high_{d-1})]
    prefix : str
        Name prefix for parameters (x0, x1, ...).

    Returns
    -------
    wrapped : callable(**kwargs) -> float
    pbounds : dict[str, tuple[float, float]]
    names : list[str]
    """
    d = len(bounds_list)
    names = [f"{prefix}{i}" for i in range(d)]
    pbounds = dict(zip(names, bounds_list))

    def wrapped(**kwargs):
        # Build x in the same order as pbounds/names
        x = np.array([kwargs[n] for n in names], dtype=float)

        y = f_vec(x)

        # f_vec might return: float, 0-D array, (1,) array, or a batch (m,) array.
        # BO needs a scalar. We coerce singletons and raise for true batches.
        if isinstance(y, np.ndarray):
            y = np.squeeze(y)
            if y.shape == ():  # 0-D array
                y = float(y)
            elif y.shape == (1,):  # singleton vector
                y = float(y[0])
            else:
                # If the function produced a batch, pick one policy:
                # raise (safest) or aggregate. Here we raise to avoid silent bugs.
                raise ValueError(
                    f"Objective returned a batch of shape {y.shape}; "
                    "BayesianOptimization expects a scalar per call."
                )
        else:
            y = float(y)

        return y

    return wrapped, pbounds, names


def as_bayes_opt_2d(f_vec2d, bounds_list, prefix="x"):
    """
    Wrap a strict 2-D objective f_vec2d(X) for bayes_opt.

    Parameters
    ----------
    f_vec2d : callable
        Objective expecting a 2-D array X of shape (m, d) and returning an array
        of shape (m,) (or something squeezable to that). Example:
            def f_vec2d(X): return X[:, 0]**3 + X[:, 1] + X[:, 2]
    bounds_list : list[tuple[float, float]]
        Bounds in order for each of the d dimensions.
    prefix : str
        Parameter name prefix for kwargs: x0, x1, ...

    Returns
    -------
    wrapped : callable(**kwargs) -> float
        Callable compatible with BayesianOptimization (single-point evaluation).
    pbounds : dict[str, tuple[float, float]]
        Name→bounds mapping for bayes_opt.
    names : list[str]
        The ordered parameter names used by wrapped.
    """
    d = len(bounds_list)
    names = [f"{prefix}{i}" for i in range(d)]
    pbounds = dict(zip(names, bounds_list))

    def wrapped(**kwargs):
        # Build a single row X of shape (1, d) preserving 'names' order
        X = np.array([kwargs[n] for n in names], dtype=float)[None, :]  # (1, d)
        y = f_vec2d(X)

        # Coerce the result to a scalar float
        if isinstance(y, np.ndarray):
            y = np.squeeze(y)
            if y.shape == ():  # 0-D array
                return float(y)
            if y.shape == (1,):  # singleton vector
                return float(y[0])
            # Anything else implies the objective returned a true batch
            raise ValueError(
                f"Expected a single value back; got array with shape {y.shape}."
            )
        # non-array return (e.g., python float) is fine
        return float(y)

    return wrapped, pbounds, names


def as_bayes_opt_auto(f_any, bounds_list, prefix="x", prefer_2d=False):
    """
    Wrap an objective that might accept either a 1-D vector x (shape (d,))
    or a strict 2-D matrix X (shape (m, d), here m=1 per BO call).

    The wrapper auto-detects the expected shape on the *first* call by trying
    one shape then the other, caches the mode, and always returns a scalar float.

    Parameters
    ----------
    f_any : callable
        Objective. Examples:
          - def f_vec(x: np.ndarray): return x[0]**3 + x[1] + x[2]        # expects (d,)
          - def f_mat(X: np.ndarray): return X[:,0]**3 + X[:,1] + X[:,2]  # expects (m,d)->(m,)
    bounds_list : list[tuple[float, float]]
        [(low_0, high_0), ..., (low_{d-1}, high_{d-1})]
    prefix : str
        Name prefix for kwargs (e.g., x0, x1, ...).
    prefer_2d : bool
        If True, try (1, d) first, else try (d,) first. Useful when you *expect*
        strict 2-D but still want auto fallback.

    Returns
    -------
    wrapped : callable(**kwargs) -> float
    pbounds : dict[str, tuple[float, float]]
    names : list[str]
    """
    d = len(bounds_list)
    names = [f"{prefix}{i}" for i in range(d)]
    pbounds = dict(zip(names, bounds_list))

    mode = {"shape": None}  # None | "1d" | "2d"

    def _coerce_scalar(y):
        # Accept float/int, 0-D np.array, or length-1 arrays
        if isinstance(y, np.ndarray):
            y = np.squeeze(y)
            if y.shape == ():  # 0-D array
                return float(y)
            if y.shape == (1,):  # singleton vector
                return float(y[0])
            # If user returned a batch with >1, that's incompatible for single-call BO
            raise ValueError(
                f"Objective returned a batch with shape {y.shape}; expected a single value."
            )
        return float(y)

    def wrapped(**kwargs):
        # keep a stable parameter order
        vec = np.array([kwargs[n] for n in names], dtype=float)  # shape (d,)

        if mode["shape"] is None:
            # Try preferred shape first, then fallback
            trials = (
                (("2d", vec[None, :]), ("1d", vec))
                if prefer_2d
                else (("1d", vec), ("2d", vec[None, :]))
            )
            last_err = None
            for shape_tag, arg in trials:
                try:
                    y = f_any(arg)
                    val = _coerce_scalar(y)
                    mode["shape"] = shape_tag
                    return val
                except Exception as e:
                    last_err = e
                    continue
            # Both attempts failed
            raise TypeError(
                "Auto-detection failed: the objective did not accept either a 1-D vector "
                f"(shape ({d},)) or a 2-D single-row matrix (shape (1, {d})). "
                f"Last error: {last_err!r}"
            )

        # Fast path: use the detected mode
        if mode["shape"] == "1d":
            y = f_any(vec)  # (d,)
        else:
            y = f_any(vec[None, :])  # (1, d)
        return _coerce_scalar(y)

    return wrapped, pbounds, names
