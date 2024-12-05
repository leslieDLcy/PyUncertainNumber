import numpy as np
from typing import Callable
from scipy.optimize import minimize

def local_optimisation_method(x: np.ndarray, f: Callable, x0: np.ndarray = None,
                              tol_loc: np.ndarray = None, options_loc: dict = None,
                              *, method_loc='Nelder-Mead'):
    """Performs local optimization to find the minimum and maximum of the function.

    This function uses `scipy.optimize.minimize` with the specified method and options 
    to find both the minimum and maximum values of the given function within the 
    specified bounds. It handles maximization by negating the function for the 
    maximization step.

    args:
        x (np.ndarray): A 2D NumPy array where each row represents an input variable and 
                            the two columns define its lower and upper bounds (interval).
        f (Callable): The objective function to be optimized. It should take a 1D NumPy array 
                        as input and return a scalar value.
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
        local_optimisation_method(x:np.ndarray, f:Callable, 
                              *, x0:np.ndarray = None,  
                              tol_loc:np.ndarray = None, options_loc: dict = None, method_loc = 'Nelder-Mead') -> dict

    note:
        - Performs local optimization to find both the minimum and maximum values of a given function, within specified bounds.
        - This function utilizes the `scipy.optimize.minimize` function to perform local optimization. 
        - Refer to `scipy.optimize.minimize` documentation for available options
   
    returns:
        dict: A dictionary containing the optimization results:
              - 'min': Results for minimization, including 'x', 'f', 'message', 'nit', 'nfev', 'final_simplex'.
              - 'max': Results for maximization, including 'x', 'f', 'message', 'nit', 'nfev', 'final_simplex'.
    example:
        # Example function 
        f = lambda x: x[0] + x[1] + x[2]  # Example function
        x_bounds = np.array([[1, 2], [3, 4], [5, 6]])

        # Initial guess (same for min and max)
        x0 = np.array([1.5, 3.5, 5.5])  

        # Different tolerances for min and max
        tol_loc = np.array([1e-4, 1e-6])  

        # Different options for min and max
        options_loc = [
            {'maxiter': 100},  # Options for minimization
            {'maxiter': 1000}  # Options for maximization
            ]

        # Perform optimization
        results = local_optimisation_method(x_bounds, f, x0=x0, tol_loc=tol_loc, 
                                            options_loc=options_loc)

        # Print the results
        print("Minimum:")
        print("  x:", results['min']['x'])
        print("  f:", results['min']['f'])
        print("  message:", results['min']['message_miny'])

        print("Maximum:")
        print("  x:", results['max']['x'])
        print("  f:", results['max']['f'])
        print("  message:", results['max']['message_maxy'])
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
    min_y = minimize(f, x0=x0[0, :], method=method_loc, bounds=bounds, tol=tol_min, options=options_min)
    max_y = minimize(negated_f, x0=x0[1, :], method=method_loc, bounds=bounds, tol=tol_max, options=options_max)
    max_y.fun = -max_y.fun  # Correct the sign of the maximum value

    # Store results in a dictionary
    results = {
        'min': {
            'x': min_y.x, #The input values that resulted in the minimum value.
            'f': min_y.fun, #The estimated minimum value of the function.
            'message_miny': min_y.message, #The success message from the minimization optimization.
            'niterations': min_y.nit, #The number of iterations for minimisation.
            'nfevaluations': min_y.nfev, #The number of function evaluations for minimisation.
            'final_simplex': min_y.final_simplex #The final simplex for minimization.
        },
        'max': {
            'x': max_y.x, #The estimated maximum value of the function.
            'f': max_y.fun, #The estimated maximum value of the function.
            'message_maxy': max_y.message, #The success message from the maximization optimization.
            'niterations': max_y.nit, #The number of iterations for maximisation.
            'nfevalutations': max_y.nfev, #The number of function evaluations for maximisation.
            'final_simplex': max_y.final_simplex #The number of function evaluations for maximization.
        }
    }

    return results

