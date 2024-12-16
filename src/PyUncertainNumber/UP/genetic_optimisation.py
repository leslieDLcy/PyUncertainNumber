import numpy as np
from typing import Callable

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback

def a(x): return np.asarray(x,dtype=float)

def genetic_optimisation_method(x_bounds: np.ndarray, f: Callable, results:dict = None,
                                 pop_size=1000, n_gen=100, tol=1e-3,
                                 n_gen_last=10, algorithm_type="NSGA2"):
    """
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
        genetic_optimisation_method(x_bounds: np.ndarray, f: Callable, results:dict,
                                    pop_size=1000, n_gen=100, tol=1e-3,
                                    n_gen_last=10, algorithm_type="NSGA2") -> dict

    note:
        Performs both minimisation and maximisation using a genetic algorithm.

    returns:
        dict: A dictionary containing the optimisation results:
            - 'bounds': An np.ndarray of the bounds for the output parameter (if f is not None). 
            - 'min': A dictionary with keys 'x', 'f', 'n_gen', and 'n_iter' for minimisation results.
            - 'max': A dictionary with keys 'x', 'f', 'n_gen', and 'n_iter' for maximisation results.
    
    example:
        # Example usage with different parameters for minimisation and maximisation
        f = lambda x: x[0] + x[1] + x[2] # Example function
        x_bounds = np.array([[1, 2], [3, 4], [5, 6]])

        # Different population sizes for min and max
        pop_size = np.array([500, 1500])  

        # Different number of generations
        n_gen = np.array([50, 150])     

        # Different tolerances
        tol = np.array([1e-2, 1e-4])     

        # Different algorithms
        algorithm_type = np.array(["GA", "NSGA2"])  

        y = genetic_optimisation_method(x_bounds, f, pop_size=pop_size, n_gen=n_gen,
                                        tol=tol, n_gen_last=10, algorithm_type=algorithm_type)

        # Print the results                                               
        print("-" * 30)
        print("bounds:", y['bounds'])

        print("Minimum:")
        print("Optimized x:", y['min']['x'])
        print("Optimized f:", y['min']['f'])
        print("Number of generations:", y['min']['n_gen'])
        print("Number of iterations:", y['min']['n_iter'])

        print("-" * 30)
        print("Maximum:")
        print("Optimized x:", y['max']['x'])
        print("Optimized f:", y['max']['f'])
        print("Number of generations:", y['max']['n_gen'])
        print("Number of iterations:", y['max']['n_iter'])

    """
    if results is None:
        results = {
             'un': None,
           
            'raw_data': {                
                'x': None,
                'f': None,
                'min': {
                        'x': None,
                        'f': None,
                        'n_gen': None,
                        'n_iter': None
                    },
                'max': {
                        'x': None,
                        'f': None,
                        'n_gen': None,
                        'n_iter': None
                    },
                'bounds': None
                }
            }
        
    class ProblemWrapper(Problem):
        def __init__(self, objective, **kwargs):
            super().__init__(n_obj=1, **kwargs)
            self.n_evals = 0
            self.objective = objective  # Store the objective ('min' or 'max')

        def _evaluate(self, x, out, *args, **kwargs):
            self.n_evals += len(x)

            # Evaluate the objective function for each individual separately
            res = np.array([f(x[i]) for i in range(len(x))])  # Apply f to each row of x

            out["F"] = res if self.objective == 'min' else -res

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
            self.history.append(algorithm.pop.get("F").min())  # Store best objective value

            # Check for convergence if enough values are available
            if len(self.history) >= self.n_last:
                last_values = self.history[-self.n_last:]
                # Calculate the range of the last 'n_last' values
                convergence_value = np.max(last_values) - np.min(last_values)
                if convergence_value <= self.tol and not self.convergence_reached:
                    self.convergence_message = "Convergence reached!"  # Store the message
                    print(self.convergence_message)
                    self.convergence_reached = True  # Set the flag to True
                    algorithm.termination.force_termination = True

    def run_optimisation(objective, pop_size, n_gen, tol, n_gen_last, algorithm_type):
        callback = ConvergenceMonitor(tol=tol, n_last=n_gen_last)
        problem = ProblemWrapper(objective=objective, n_var=x_bounds.shape[0],
                                  xl=x_bounds[:, 0], xu=x_bounds[:, 1])
        algorithm = GA(pop_size=pop_size) if algorithm_type == "GA" else NSGA2(pop_size=pop_size)
        result = minimize(problem, algorithm, ('n_gen', n_gen), callback=callback)
        return result, callback.n_generations, problem.n_evals, callback.convergence_message

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
        'min', pop_size[0], n_gen[0], tol[0], n_gen_last[0], algorithm_type[0]
    )

    # --- Maximisation ---
    result_max, n_gen_max, n_iter_max, message_max = run_optimisation(
        'max', pop_size[1], n_gen[1], tol[1], n_gen_last[1], algorithm_type[1]
    )

    # Create a dictionary to store the results
    results = {
        'un': None,
        
        'raw_data': {
            'min': {
                'x': result_min.X,
                'f': result_min.F,
                'message' : message_min,
                'n_gen': n_gen_min,
                'n_iter': n_iter_min
                },
            'max': {
                'x': result_max.X,
                'f': -result_max.F,  # Negate the result for maximisation
                'message' : message_max,
                'n_gen': n_gen_max,
                'n_iter': n_iter_max
                },
            'bounds' : np.array([result_min.F[0], -result_max.F[0]])
            }
        }

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
# print("-" * 30)
# print("bounds:", y['raw_data']['bounds'])

# print("Minimum:")
# print("Optimized x:", y['raw_data']['min']['x'])
# print("Optimized f:", y['raw_data']['min']['f'])
# print("Number of generations:", y['raw_data']['min']['n_gen'])
# print("Number of iterations:", y['raw_data']['min']['n_iter'])

# print("-" * 30)
# print("Maximum:")
# print("Optimized x:", y['raw_data']['max']['x'])
# print("Optimized f:", y['raw_data']['max']['f'])
# # print("Number of generations:", y['raw_data']['max']['n_gen'])
# # print("Number of iterations:", y['raw_data']['max']['n_iter'])