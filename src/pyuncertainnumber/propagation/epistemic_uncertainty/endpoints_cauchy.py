import numpy as np
import tqdm
from typing import Callable, Union
from scipy.optimize import brentq
from pyuncertainnumber.propagation.utils import Propagation_results


def cauchydeviates_method(
    x: np.ndarray,
    f: Callable,
    results: Propagation_results = None,
    n_sam: int = 500,
    xtol: Union[float, np.ndarray]  = 2e-12,
    rtol: Union[float, np.ndarray]  = 8.881784197001252e-16,
    maxiter: Union[float, np.ndarray]  = 100,
    min_y0: Union[float, np.ndarray] = 1e-6,   
    save_raw_data="no",
) -> Propagation_results:  # Specify return type
    """This method propagates intervals through a balck box model with the Cauchy deviate method. 
    It is an approximate method, so the user should expect non-identical results for different runs.

    args:
        x (np.ndarray): A 2D NumPy array representing the intervals for each input variable.
                         Each row should contain two elements: the lower and upper bounds of the interval.
        f (Callable): A callable function that takes a 1D NumPy array of input values
                        and returns a single output value or an array of output values.
                        Can be None, in which case only the Cauchy deviates (x) and the
                        maximum Cauchy deviate (K) are returned.
        n_sam (int): The number of samples (Cauchy deviates) to generate for each input variable (defaults 500 samples).
        results (Propagation_results): The class to use for storing results (defaults to Propagation_results).
        xtol (float): The absolute tolerance(s) for the root-finding algorithm (brentq). 
                      If a float, the same xtol is used for all outputs.If an np.ndarray, it should have the same length 
                      as the output of f, providing an absolute tolerance for brentq for each output. Default: 2e-12.
        rtol (float): The relative tolerance(s) for the root-finding algorithm (brentq). 
                      If a float, the same rtol is used for all outputs.If an np.ndarray, it should have the same length 
                      as the output of f, providing a relative tolerance for brentq for each output. Default: 8.88e-16.
        maxiter (int): The maximum number(s) of iterations for the root-finding algorithm (brentq). 
                       If a float, the same maxiter is used for all outputs.If an np.ndarray, it should have the same length 
                       as the output of f, providing a maximum number of iterations for brentq for each output. Default: 100.
        min_y0 (float, or np.ndarray): The lower bound(s) for the interval(s) in the brentq root-finding algorithm.
                                       If a float, the same lower bound is used for all outputs.
                                       If an np.ndarray, it should have the same length as the output of f,
                                       providing a separate lower bound for each output. Default: 1e-6.
        save_raw_data (str, optional): Acts as a switch to enable or disable the storage of raw input data when a function (f) 
          is not provided.
          - 'no': Returns an error that no function is provided.
          - 'yes': Returns the full arrays of unique input combinations.

    signature:
        cauchydeviate_method(x: np.ndarray, f: Callable, results_class = Propagation_results,
                        n_sam: int, save_raw_data='no') -> Propagation_results

    notes:
        The interval for the `brentq` root-finding algorithm is chosen heuristically.
        The lower bound is set to 1e-6, assuming the root (representing the change
        in the function's value) is positive and not exactly zero.  The upper bound
        is calculated as half the maximum absolute difference observed between the
        function's value at the central point and the values at the Cauchy deviate
        samples (`max(filtered_deltaF) / 2` or `max(filtered_deltaF_i) / 2`).
        This choice is based on the assumption that the actual change in the 
        function's value is likely to be smaller than the largest observed change.

        Users should be aware that these choices are heuristics and might need to be
        adjusted depending on the specific function and problem being solved. If
        `brentq` fails to converge, consider experimenting with different interval
        bounds.  For example, if the function is highly sensitive to input changes,
        the upper bound might be too large, and dividing by a larger number (e.g.,
        4 or 8 instead of 2) might be necessary.  Alternatively, if the root is
        expected to be very close to zero, the lower bound might need to be
        reduced.

    returns:
        A PropagationResult object containing the results.
            - 'un': UncertainNumber object(s) representing the uncertainty in the output(s) of the function.
            - 'raw_data': A dictionary containing raw data (if f is None):
                - 'x': Cauchy deviates (x).
                - 'K': Maximum Cauchy deviate (K).
                - 'bounds': An np.ndarray of the bounds for each output parameter (if f is not None).
                - 'min': A dictionary for lower bound results (if f is not None).
                    - 'f': Minimum output value(s).
                    - 'x': None (as input values corresponding to min/max are not tracked in this method).
                - 'max':  A dictionary for upper bound results (if f is not None).
                    - 'f': Maximum output value(s).
                    - 'x': None (as input values corresponding to min/max are not tracked in this method).

    raises:
        ValueError if no function is provided and save_raw_data is 'no'.

    example:
        >>> f = lambda x: x[0] + x[1] + x[2]  # Example function
        >>> x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = cauchydeviates_method(x_bounds,f=f, n_sam=50, save_raw_data = 'yes')
    """
    print(
        f"Total number of input combinations for the endpoints Cauchy deviates method: {n_sam}"
    )

    if results is None:
        results = Propagation_results()  # Create an instance of Propagation_results

    x = np.atleast_2d(x)  # Ensure x is 2D
    lo, hi = x.T  # Unpack lower and upper bounds directly

    xtilde = (lo + hi) / 2
    Delta = (hi - lo) / 2

    if f is not None:  # Only evaluate if f is provided
        ytilde = f(xtilde)

        if isinstance(ytilde, (float, np.floating)):  # Handle scalar output
            deltaF = np.zeros(n_sam)
            for k in tqdm.tqdm(range(1, n_sam), desc="Calculating Cauchy deviates"):
                r = np.random.rand(x.shape[0])
                c = np.tan(np.pi * (r - 0.5))
                K = np.max(c)
                delta = Delta * c / K
                x_sample = xtilde - delta
                deltaF[k] = K * (ytilde - f(x_sample))

            def Z(Del):
                return n_sam / 2 - np.sum(1 / (1 + (deltaF / Del) ** 2))
            
            mask = np.isnan(deltaF)
            filtered_deltaF = deltaF[~mask]

            # Use a small value for the lower bound
            zRoot = brentq(Z, min_y0, max(filtered_deltaF) / 2, 
                                              xtol =  xtol, rtol = rtol,maxiter=maxiter )
            min_candidate = np.array([ytilde - zRoot])
            max_candidate = np.array([ytilde + zRoot])
            bounds = np.array([min_candidate[0], max_candidate[0]])

            results.raw_data["min"] = np.array([{"x": None, "f": min_candidate}])
            results.raw_data["max"] = np.array([{"x": None, "f": max_candidate}])
            results.raw_data["bounds"] = bounds

        else:  # Handle array output
            len_y = len(ytilde)
            deltaF = np.zeros((n_sam, len_y))
            min_candidate = np.empty(len_y)
            max_candidate = np.empty(len_y)

            if isinstance(min_y0, float):
                min_y0_arr = np.full(len_y, min_y0)
            else:
                min_y0_arr = min_y0
            
            if isinstance(xtol, float):
                xtol_arr = np.full(len_y, xtol)
            else:
                xtol_arr = xtol
            
            if isinstance(rtol, float):
                rtol_arr = np.full(len_y, rtol)
            else:
                rtol_arr = rtol
            
            if isinstance(maxiter, float):
                maxiter_arr = np.full(len_y, maxiter)
            else:
                maxiter_arr = maxiter


            for k in tqdm.tqdm(range(n_sam), desc="Calculating Cauchy deviates"):
                r = np.random.rand(x.shape[0])
                c = np.tan(np.pi * (r - 0.5))
                K = np.max(c)
                delta = Delta * c / K
                x_sample = xtilde - delta
                result = f(x_sample)
                for i in range(len_y):
                    deltaF[k, i] = K * (ytilde[i] - result[i])

            bounds = np.zeros((len_y, 2))  # Initialize bounds array
            for i in range(len_y):
                mask = np.isnan(deltaF[:, i])
                filtered_deltaF_i = deltaF[:, i][~mask]

                def Z(Del):
                    return n_sam / 2 - np.sum(1 / (1 + (filtered_deltaF_i / Del) ** 2))

                try:  # Handle potential errors in brentq
                    # Use a small value for the lower bound
                    zRoot = brentq(Z, min_y0_arr[i], max(filtered_deltaF_i) / 2, 
                                              xtol =  xtol_arr[i], rtol = rtol_arr[i],maxiter=maxiter_arr[i] )
                except ValueError:
                    print(f"Warning: brentq failed for output {i}. Using 0 for zRoot.")
                    zRoot = 0  # Or handle the error in another way
                min_candidate[i] = ytilde[i] - zRoot
                max_candidate[i] = ytilde[i] + zRoot
                bounds[i] = [min_candidate[i], max_candidate[i]]

            # Populate the results object
            results.raw_data["bounds"] = bounds
            results.raw_data["min"] = np.append(
                results.raw_data["min"],
                [{"x": None, "f": min_val} for min_val in min_candidate],
            )
            results.raw_data["max"] = np.append(
                results.raw_data["max"],
                [{"x": None, "f": max_val} for max_val in max_candidate],
            )

        print("Input x for min max y are NOT available for the Cauchy method!")

        if save_raw_data == "yes":
            print("Input-Output raw data are NOT available for the Cauchy method!")

    elif save_raw_data == "yes":  # If f is None and save_raw_data is 'yes'
        x_samples = np.zeros((n_sam, x.shape[0]))
        K_values = np.zeros(n_sam)
        for k in tqdm.tqdm(range(n_sam), desc="Calculating Cauchy deviates"):
            r = np.random.rand(x.shape[0])
            c = np.tan(np.pi * (r - 0.5))
            K = np.max(c)
            delta = Delta * c / K
            x_samples[k] = xtilde - delta
            K_values[k] = K

        results.add_raw_data(x=x_samples, K=K_values)

    else:
        print(
            "No function is provided. Select save_raw_data = 'yes' to save the input combinations"
        )

    return results


# # # Example usage with different parameters for minimization and maximization
# f = lambda x: x[0] + x[1] + x[2]  # Example function

# # Determine input parameters for function and method
# x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
# n=50
# # Call the method
# y = cauchydeviates_method(x_bounds,f=f, n_sam=n, save_raw_data= 'yes')
# # # print the results
# # y.summary()
