from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import numpy as np
import inspect
from functools import partial
import warnings


class BayesOpt:
    """Bayesian Optimization class for the uncertain number"""

    # TODO add a descriptor for `task`
    def __init__(
        self,
        f,
        xc_bounds,
        dimension,
        task="maximisation",
        num_explorations=100,
        num_iterations=100,
        acquisition_function=None,
    ):

        self.task = task  # either minimisation or maximisation
        self.num_explorations = num_explorations  # initial exploration points
        self.num_iterations = num_iterations
        self.xc_bounds = xc_bounds
        self.dimension = dimension
        self.acquisition_function = (
            acquisition_function
            if acquisition_function
            else acquisition.UpperConfidenceBound(kappa=10.0)
        )
        self.f = f  # the function to be optimised

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
    def optimal(self):
        return self._optimal_dict

    @property
    def optimal_xc(self):
        """return the optimal xc"""
        return self._optimal_dict["xc"]

    @property
    def optimal_target(self):
        """return the optimal target"""
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
