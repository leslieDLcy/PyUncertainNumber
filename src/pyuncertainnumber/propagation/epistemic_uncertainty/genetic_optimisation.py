
import numpy as np
from typing import Callable
from typing import Callable, Union
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pyuncertainnumber.propagation.utils import Propagation_results
#TODO need to add it to final, bug removed. 
def genetic_optimisation_method(x_bounds:  np.ndarray,  # Updated to list of lists
                                f: Callable = None,
                                results: Propagation_results = None,
                                pop_size: Union[int, np.ndarray] = 1500,
                                n_gen: Union[int, np.ndarray]= 150,
                                tol: Union[int, np.ndarray] = 1e-3,
                                n_gen_last: Union[int, np.ndarray] = 10
                                ) -> Propagation_results:
    """
    Performs both minimisation and maximisation using a genetic algorithm.

    args:
        x_bounds (np.ndarray): Bounds for decision variables.
        f (Callable): Objective function to optimize.
        results (Propagation_results, optional): Existing results object. Defaults to None.
        pop_size (int, list, or np.ndarray): Population size.
        n_gen (int, list, or np.ndarray): Maximum number of generations.
        tol (float, list, or np.ndarray): Tolerance for convergence check.
        n_gen_last (int, list, or np.ndarray): Number of last generations to consider for convergence.
       
    returns:
        Propagation_results: Object containing optimisation results.
    """
    class ProblemWrapper(Problem):
        """Wraps the objective function for pymoo."""
        def __init__(self, objective, **kwargs):
            super().__init__(n_obj=1, **kwargs)
            self.n_evals = 0
            self.objective = objective

        def _evaluate(self, x, out, *args, **kwargs):
            """Evaluates the objective function."""
            self.n_evals += len(x)
            out["F"] = np.array([f(xi) for xi in x]) if self.objective == 'min' else -np.array([f(xi) for xi in x])

    class ConvergenceMonitor(Callback):
        """Monitors convergence of the genetic algorithm."""
        def __init__(self, tol=1e-4, n_last=5):
            super().__init__()
            self.tol = tol
            self.n_last = n_last
            self.history = []
            self.n_generations = 0
            self.convergence_reached = False
            self.convergence_message = None

        def notify(self, algorithm):
            """Checks for convergence and updates history."""
            self.n_generations += 1
            self.history.append(algorithm.pop.get("F").min())
            if len(self.history) >= self.n_last:
                last_values = self.history[-self.n_last:]
                convergence_value = np.max(last_values) - np.min(last_values)
                if convergence_value <= self.tol and not self.convergence_reached:
                    self.convergence_message = "Convergence reached!"
                    print(self.convergence_message)
                    self.convergence_reached = True
                    algorithm.termination.force_termination = True

    def run_optimisation(objective, pop_size, n_gen, tol, n_gen_last):
        """Runs the optimisation algorithm."""
        callback = ConvergenceMonitor(tol=tol, n_last=n_gen_last)
        problem = ProblemWrapper(objective=objective, n_var=x_bounds.shape[0],
                                    xl=x_bounds[:, 0], xu=x_bounds[:, 1])
        algorithm = GA(pop_size=pop_size) #force GA algorithm
        result = minimize(problem, algorithm, ('n_gen', n_gen), callback=callback)
        return result, callback.n_generations, problem.n_evals, callback.convergence_message

    def handle_arg(arg):
        """Handles arguments that can be single values or lists."""
        if isinstance(arg, str):
            return [arg, arg]
        elif isinstance(arg, np.ndarray):
            return [int(a) for a in arg]
        elif isinstance(arg, list):
            return [int(a) for a in arg]
        else:
            return [int(arg), int(arg)]

    pop_size = handle_arg(pop_size)
    n_gen = handle_arg(n_gen)
    tol = handle_arg(tol)
    n_gen_last = handle_arg(n_gen_last)

    result_min, n_gen_min, n_iter_min, message_min = run_optimisation(
        'min', list(pop_size)[0], list(n_gen)[0], list(tol)[0], list(n_gen_last)[0]
    )

    result_max, n_gen_max, n_iter_max, message_max = run_optimisation(
        'max', list(pop_size)[1], list(n_gen)[1], list(tol)[1], list(n_gen_last)[1]
    )

    if results is None:
        results = Propagation_results()

    if not hasattr(results, 'raw_data') or results.raw_data is None:
        results.raw_data = {'min': [], 'max': []}

    results.raw_data['min']= np.append(results.raw_data['min'], {
        'x': result_min.X,
        'f': result_min.F[0],
        'message': message_min,
        'ngenerations': n_gen_min,
        'niterations': n_iter_min
    })
    
    results.raw_data['max'] = np.append(results.raw_data['max'], {
        'x': result_max.X,
        'f':  - result_max.F[0],
        'message': message_max,
        'ngenerations': n_gen_max,
        'niterations': n_iter_max
    })

    results.raw_data['bounds'] = np.array([results.raw_data['min'][0]['f'],results.raw_data['max'][0]['f']])

    return results


# # Example usage with different parameters for minimisation and maximisation
# f = lambda x: x[0] + x[1] + x[2] # Example function
# x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
# pop_size = np.array([1000, 1500])
# n_gen = np.array([100, 150])
# tol = np.array([1e-4, 1e-4])
# algorithm_type = "GA"

# y = genetic_optimisation_method(x_bounds, f, pop_size=pop_size, n_gen=n_gen,
#                                 tol=tol, n_gen_last=10, algorithm_type=algorithm_type)

# print('max_message', y.raw_data['max'][0]['message'])
# print('min_message',y.raw_data['min'][0]['message'])