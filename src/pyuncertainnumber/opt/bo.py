from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import numpy as np
import inspect
from functools import partial
import warnings


class BayesOpt:
    """Bayesian Optimisation class

    args:
        f (callable): the target function to be optimised, should have a single argument

        xc_bounds (dict): the bounds for the design space, e.g. {'x1': (0, 1), 'x2': (0, 1)}

        dimension (int): the dimension of the design space, i.e. the number of parameters

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

    example:
        >>> import numpy as np
        >>> from pyuncertainnumber.opt.bo import BayesOpt
        >>> def black_box_function(x):
        ...     return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2 / 10) + 1 / (x**2 + 1)
        >>> bo = BayesOpt(
        ...     f=black_box_function,
        ...     dimension=1,
        ...     xc_bounds={'x': (-2, 10)},
        ...     task='maximisation',
        ...     num_explorations=3,
        ...     num_iterations=20
        ... )
        >>> bo.run(verbose=True)
        >>> print(bo.optimal)  # get the optimal parameters and target value

    .. admonition:: Implementation

        For 2D (or higher) optimisation task, the variable bounds can be specified using `EpistemicDomain` class.

        example:
            >>> from from pyuncertainnumber import pba, EpistemicDomain
            >>> ed = EpistemicDomain(pba.I(-5, 5), pba.I(-5, 5))
            >>> BayesOpt(f=foo,
            ...     dimension=2,
            ...     xc_bounds= ed.to_BayesOptBounds(),  # the trick
            ...     task='maximisation',
            ...     num_explorations=3,
            ...     num_iterations=20,
            ...     )
    """

    # TODO add a descriptor for `task`
    def __init__(
        self,
        f,
        xc_bounds,
        dimension,
        task,
        acquisition_function="UCB",
        num_explorations=100,
        num_iterations=100,
    ):

        self.task = task  # either minimisation or maximisation
        self.num_explorations = num_explorations  # initial exploration points
        self.num_iterations = num_iterations
        self.xc_bounds = xc_bounds
        self.dimension = dimension
        self.acquisition_function = self.parse_acq(acquisition_function)
        self.f = f  # the function to be optimised

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
        # step 1: dimension check
        if not check_argument_count(f) == "Single argument":
            warnings.warn(
                "The function to be optimised should have a single argument",
                category=RuntimeWarning,
            )
        if self.dimension > 1 and (check_argument_count(f) == "Single argument"):
            f = partial(transform_func, fb=f)
        # step 2: flip check by the task
        # the first flip is to make the function minimisation
        if self.task == "maximisation":
            self._f = f
        elif self.task == "minimisation":
            self._f = lambda *args, **kwargs: -f(*args, **kwargs)

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
            pbounds=self.xc_bounds,
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
