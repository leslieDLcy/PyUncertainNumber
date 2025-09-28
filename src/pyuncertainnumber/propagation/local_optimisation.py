import numpy as np
from typing import Callable
from scipy.optimize import minimize
from pyuncertainnumber.propagation.utils import Propagation_results


def local_optimisation_method(x: np.ndarray, f: Callable, x0: np.ndarray = None,
                              results: Propagation_results = None,
                              tol_loc: np.ndarray = None, options_loc: dict = None,
                              *, method_loc='Nelder-Mead') -> Propagation_results:  # Specify return type
    """  
        Performs local optimization to find both the minimum and maximum values of a given function, within specified bounds.

    args:
        x (np.ndarray): A 2D NumPy array where each row represents an input variable and 
                        the two columns define its lower and upper bounds (interval).
        f (Callable): The objective function to be optimized. It should take a 1D NumPy array 
                      as input and return a scalar value.
        results (Propagation_results, optional): An object to store propagation results.
                                            Defaults to None, in which case a new
                                            `Propagation_results` object is created.
        x0 (np.ndarray, optional): A 1D or 2D NumPy array representing the initial guess for the 
                                    optimization. 
                                    - If x0 has shape (n,), the same initial values are used for both 
                                      minimization and maximization.
                                    - If x0 has shape (2, n), x0[0, :] is used for minimization and 
                                      x0[1, :] for maximization.
                                    If not provided, the midpoint of each variable's interval is used.
                                    Defaults to None.
        tol_loc (np.ndarray, optional): Tolerance for termination.
                                        - If tol_loc is a scalar, the same tolerance is used for both 
                                          minimization and maximization.
                                        - If tol_loc is an array of shape (2,), tol_loc[0] is used for 
                                          minimization and tol_loc[1] for maximization.
                                        Defaults to None.
        options_loc (dict, optional): A dictionary of solver options. 
                                      - If options_loc is a dictionary, the same options are used for 
                                        both minimization and maximization.
                                      - If options_loc is a list of two dictionaries, options_loc[0] 
                                        is used for minimization and options_loc[1] for maximization.
                                      Refer to `scipy.optimize.minimize` documentation for available 
                                      options. Defaults to None.
        method_loc (str, optional): The optimization method to use (e.g., 'Nelder-Mead', 'COBYLA'). 
                                    Defaults to 'Nelder-Mead'.

   signature:
        local_optimisation_method(x:np.ndarray, f:Callable,  results = None,
                              *, x0:np.ndarray = None,  
                              tol_loc:np.ndarray = None, options_loc: dict = None, method_loc = 'Nelder-Mead') -> dict

    notes:
        This function utilizes the `scipy.optimize.minimize` function to perform local optimization. 
        Refer to `scipy.optimize.minimize` documentation for available options.
        It only handles a function which produces a single output.
    
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
        >>> f = lambda x: x[0] + x[1] + x[2]  # Example function
        >>> x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
        >>> # Initial guess (same for min and max)
        >>> x0 = np.array([1.5, 3.5, 5.5])  
        >>> # Different tolerances for min and max
        >>> tol_loc = np.array([1e-4, 1e-6])  
        >>> # Different options for min and max
        >>> options_loc = [
        >>>     {'maxiter': 100},  # Options for minimization
        >>>     {'maxiter': 1000}  # Options for maximization
        >>>     ]
        >>> # Perform optimization
        >>> y = local_optimisation_method(x_bounds, f, x0=x0, tol_loc=tol_loc, 
        >>>                                     options_loc=options_loc)

    """

    bounds = [(var[0], var[1]) for var in x]

    def negated_f(x):
        return -f(x)

    if x0 is None:
        x0 = np.mean(x, axis=1)  # Use midpoint of intervals as initial guess
    x0 = np.atleast_2d(x0)  # Ensure x0 is 2D
    if x0.shape[0] == 1:  # If only one initial guess is provided, use it for both min and max
        x0 = np.tile(x0, (2, 1))

    # Handle tol_loc
    if tol_loc is None:
        tol_min = None  # Use default tolerance
        tol_max = None  # Use default tolerance
    elif np.isscalar(tol_loc):  # If tol_loc is a scalar
        tol_min = tol_loc
        tol_max = tol_loc
    else:  # If tol_loc is an array
        tol_min = tol_loc[0]
        tol_max = tol_loc[1]

    #  Handle options_loc
    if options_loc is None:
        options_min = None  # Use default options
        options_max = None  # Use default options
    elif isinstance(options_loc, dict):  # If options_loc is a single dictionary
        options_min = options_loc
        options_max = options_loc
    else:  # If options_loc is a list of two dictionaries
        options_min = options_loc[0]
        options_max = options_loc[1]

    # Perform minimization and maximization
    min_y = minimize(f, x0=x0[0, :], method=method_loc,
                     bounds=bounds, tol=tol_min, options=options_min)
    max_y = minimize(negated_f, x0=x0[1, :], method=method_loc,
                     bounds=bounds, tol=tol_max, options=options_max)
    max_y.fun = -max_y.fun  # Correct the sign of the maximum value

    # Create an instance of the result class
    if results is None:
        results = Propagation_results()  # Create an instance of Propagation_results

    # Store ALL results in the results object with descriptions
    results.raw_data['min'] = np.append(results.raw_data['min'], {'x': min_y.x,
                                                                  'f': min_y.fun,
                                                                  'message': min_y.message,
                                                                  'niterations': min_y.nit,
                                                                  'nfevaluations': min_y.nfev,
                                                                  'final_simplex': min_y.final_simplex
                                                                  })

    results.raw_data['max'] = np.append(results.raw_data['max'], {
        'x': max_y.x,
        'f': max_y.fun,
        'message': max_y.message,
        'niterations': max_y.nit,
        'nfevaluations': max_y.nfev,
        'final_simplex': max_y.final_simplex
    })

    results.raw_data['bounds'] = np.array([min_y.fun, max_y.fun])

    return results

# # Example function
# f = lambda x: x[0] + x[1] + x[2]  # Example function
# x_bounds = np.array([[1, 2], [3, 4], [5, 6]])

# # Initial guess (same for min and max)
# x0 = np.array([1.5, 3.5, 5.5])

# # Different tolerances for min and max
# tol_loc = np.array([1e-4, 1e-6])

# # Different options for min and max
# options_loc = [
#     {'maxiter': 100},  # Options for minimization
#     {'maxiter': 1000}  # Options for maximization
#      ]

# # Perform optimization
# y = local_optimisation_method(x_bounds, f, x0=x0, tol_loc=tol_loc,
#                                 options_loc=options_loc)

# # Print the results
# y.print()
