from __future__ import annotations
from typing import TYPE_CHECKING
from ...pba.intervals.number import Interval

import numpy as np
from typing import Callable
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pyuncertainnumber.propagation.utils import Propagation_results


def a(x):
    return np.asarray(x, dtype=float)


def genetic_optimisation_method(
    x_bounds: Interval | np.ndarray,
    f: Callable,
    results: Propagation_results = None,
    pop_size=1000,
    n_gen=100,
    tol=1e-3,
    n_gen_last=10,
    algorithm_type="NSGA2",
) -> Propagation_results:  # Specify return type
    """Performs both minimisation and maximisation using a genetic algorithm.

    args:
        x_bounds: Bounds for decision variables (NumPy array).
        f: Objective function to optimize.
        pop_size: Population size (int or array of shape (2,)).
        n_gen: Maximum number of generations (int or array of shape (2,)).
        tol: Tolerance for convergence check (float or array of shape (2,)).
        n_gen_last: Number of last generations to consider for convergence
                    (int or array of shape (2,)).
        algorithm_type: 'NSGA2' or 'GA' to select the optimisation algorithm
                        (str or array of shape (2,)).

    signature:
        genetic_optimisation_method(x_bounds: np.ndarray, f: Callable,
                                    pop_size=1000, n_gen=100, tol=1e-3,
                                    n_gen_last=10, algorithm_type="NSGA2") -> Propagation_results


    notes:
        It only handles a function which produces a single output.
        Refer to `pymoo.optimize` documentation for available options.

    returns:
        An `Propagation_results` object which contains:
            - 'un': UncertainNumber object(s) to represent the interval of the output.
            - 'raw_data' (dict): Dictionary containing raw data shared across output:
                    - 'x' (np.ndarray): Input values.
                    - 'f' (np.ndarray): Output values.
                    - 'min' (np.ndarray): Array of dictionaries for the function's output,
                              containing 'f' for the minimum of that output as well 'message', 'nit', 'nfev', 'final_simplex'.
                    - 'max' (np.ndarray): Array of dictionaries for the function's output,
                              containing 'f' for the maximum of that output as well 'message', 'nit', 'nfev', 'final_simplex'.
                    - 'bounds' (np.ndarray): 2D array of lower and upper bounds for the output.

    example:
        >>> # Example usage with different parameters for minimisation and maximisation
        >>> f = lambda x: x[0] + x[1] + x[2] # Example function
        >>> x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
        >>> # Different population sizes for min and max
        >>> pop_size = np.array([500, 1500])
        >>> # Different number of generations
        >>> n_gen = np.array([50, 150])
        >>> # Different tolerances
        >>> tol = np.array([1e-2, 1e-4])
        >>> # Different algorithms
        >>> algorithm_type = np.array(["GA", "NSGA2"])
        >>> y = genetic_optimisation_method(x_bounds, f, pop_size=pop_size, n_gen=n_gen,
        >>>                                 tol=tol, n_gen_last=10, algorithm_type=algorithm_type)

    """

    if isinstance(x, Interval):
        x = x.to_numpy()

    # TODO move class definintion outside the function @ioannaioan
    class ProblemWrapper(Problem):
        def __init__(self, objective, **kwargs):
            super().__init__(n_obj=1, **kwargs)
            self.n_evals = 0
            self.objective = objective  # Store the objective ('min' or 'max')

        def _evaluate(self, x, out, *args, **kwargs):
            self.n_evals += len(x)

            # Evaluate the objective function for each individual separately
            # Apply f to each row of x
            res = np.array([f(x[i]) for i in range(len(x))])

            out["F"] = res if self.objective == "min" else -res

    class ConvergenceMonitor(Callback):
        def __init__(self, tol=1e-4, n_last=5):
            super().__init__()
            self.tol = tol  # Tolerance for
            self.n_last = n_last  # Number of last values to consider
            self.history = []  # Store the history of objective values
            self.n_generations = 0
            self.convergence_reached = False  # Flag to track convergence

        def notify(self, algorithm):
            self.n_generations += 1
            # Store best objective value
            self.history.append(algorithm.pop.get("F").min())

            # Check for convergence if enough values are available
            if len(self.history) >= self.n_last:
                last_values = self.history[-self.n_last :]
                # Calculate the range of the last 'n_last' values
                convergence_value = np.max(last_values) - np.min(last_values)
                if convergence_value <= self.tol and not self.convergence_reached:
                    self.convergence_message = (
                        "Convergence reached!"  # Store the message
                    )
                    print(self.convergence_message)
                    self.convergence_reached = True  # Set the flag to True
                    algorithm.termination.force_termination = True

    def run_optimisation(objective, pop_size, n_gen, tol, n_gen_last, algorithm_type):
        callback = ConvergenceMonitor(tol=tol, n_last=n_gen_last)
        problem = ProblemWrapper(
            objective=objective,
            n_var=x_bounds.shape[0],
            xl=x_bounds[:, 0],
            xu=x_bounds[:, 1],
        )
        algorithm = (
            GA(pop_size=pop_size)
            if algorithm_type == "GA"
            else NSGA2(pop_size=pop_size)
        )
        result = minimize(problem, algorithm, ("n_gen", n_gen), callback=callback)
        return (
            result,
            callback.n_generations,
            problem.n_evals,
            callback.convergence_message,
        )

    # Handle arguments that can be single values or arrays
    def handle_arg(arg):
        return np.array([arg, arg]) if not isinstance(arg, np.ndarray) else arg

    pop_size = handle_arg(pop_size)
    n_gen = handle_arg(n_gen)
    tol = handle_arg(tol)
    n_gen_last = handle_arg(n_gen_last)
    algorithm_type = handle_arg(algorithm_type)

    # --- Minimisation ---
    result_min, n_gen_min, n_iter_min, message_min = run_optimisation(
        "min", pop_size[0], n_gen[0], tol[0], n_gen_last[0], algorithm_type[0]
    )

    # --- Maximisation ---
    result_max, n_gen_max, n_iter_max, message_max = run_optimisation(
        "max", pop_size[1], n_gen[1], tol[1], n_gen_last[1], algorithm_type[1]
    )

    # Create a dictionary to store the results
    if results is None:
        results = Propagation_results()  # Create an instance of Propagation_results

    # Store ALL results in the results object with descriptions
    results.raw_data["min"] = np.append(
        results.raw_data["min"],
        {
            "x": result_min.X,
            "f": result_min.F,
            "message": message_min,
            "ngenerations": n_gen_min,
            "niterations": n_iter_min,
        },
    )

    results.raw_data["max"] = np.append(
        results.raw_data["min"],
        {
            "x": result_max.X,
            "f": -result_max.F,  # Negate the result for maximisation
            "message": message_max,
            "ngenerations": n_gen_max,
            "niterations": n_iter_max,
        },
    )

    results.raw_data["bounds"] = np.array([result_min.F[0], -result_max.F[0]])

    return results


# # Example usage with different parameters for minimisation and maximisation
# f = lambda x: x[0] + x[1] + x[2] # Example function
# x_bounds = np.array([[1, 2], [3, 4], [5, 6]])

# # Different population sizes for min and max
# pop_size = np.array([500, 1500])

# # Different number of generations
# n_gen = np.array([50, 150])

# # Different tolerances
# tol = np.array([1e-2, 1e-4])

# # Different algorithms
# algorithm_type = np.array(["GA", "NSGA2"])

# y = genetic_optimisation_method(x_bounds, f, pop_size=pop_size, n_gen=n_gen,
#                                         tol=tol, n_gen_last=10, algorithm_type=algorithm_type)

# # Print the results
# y.print()
