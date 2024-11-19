import numpy as np
import tqdm
from typing import Callable
from scipy.optimize import brentq

def cauchydeviate_method(x: np.ndarray, f: Callable, n: int, save_raw_data='no'):
    """
    args:
        x (np.ndarray): A 2D NumPy array representing the intervals for each input variable.
                         Each row should contain two elements: the lower and upper bounds of the interval.
        f (Callable): A callable function that takes a 1D NumPy array of input values
                        and returns a single output value or an array of output values.
                        Can be None, in which case only the Cauchy deviates (x) and the 
                        maximum Cauchy deviate (K) are returned.
        n (int): The number of samples (Cauchy deviates) to generate for each input variable.
        save_raw_data (str, optional): Whether to save raw data. Defaults to 'no'.
                                        Currently not supported by this method.
    
    signature:
        cauchydeviate_method(x: np.ndarray, f: Callable, n: int, save_raw_data='no') -> dict

    note:
        - This method propagates intervals through a balck box model with the endpoint Cauchy deviate method. 
        - It is an approximate method, so the user should expect non-identical results for different runs. 

    returns:
        dict: A dictionary containing the results:
            - 'bounds': An np.ndarray of the bounds for each output parameter (if f is not None). 
            - 'min': A dictionary for lower bound results (if f is not None).
                  - 'f': Minimum output value(s).
                  - 'x': None (as input values corresponding to min/max are not tracked in this method).
              - 'max':  A dictionary for upper bound results (if f is not None).
                  - 'f': Maximum output value(s).
                  - 'x': None (as input values corresponding to min/max are not tracked in this method).
              - 'raw_data': A dictionary containing raw data (if f is None):
                  - 'x': Cauchy deviates (x).
                  - 'K': Maximum Cauchy deviate (K).
    """

    if save_raw_data != 'no':
        print("Input-Output raw data are NOT available for the Cauchy method!")

    x = np.atleast_2d(x)  # Ensure x is 2D
    lo, hi = x.T  # Unpack lower and upper bounds directly

    xtilde = (lo + hi) / 2
    Delta = (hi - lo) / 2

    results = {
        'bounds': None,
        'min': {'f': None, 'x': None},
        'max': {'f': None, 'x': None},
        'raw_data': {'x': None, 'K': None, 'f' : None}
    }  # Initialize with None values

    if f is not None:  # Only evaluate if f is provided
        ytilde = f(xtilde)

        if isinstance(ytilde, (float, np.floating)):  # Handle scalar output
            deltaF = np.zeros(n)
            for k in tqdm.tqdm(range(1, n), desc="Calculating Cauchy deviates"):
                r = np.random.rand(x.shape[0])
                c = np.tan(np.pi * (r - 0.5))
                K = np.max(c)
                delta = Delta * c / K
                x_sample = xtilde - delta
                deltaF[k] = (K * (ytilde - f(x_sample)))

            Z = lambda Del: n/2 - np.sum(1 / (1 + (deltaF / Del)**2))
            zRoot = brentq(Z, 1e-6, max(deltaF)/2)  # Use a small value for the lower bound
            min_candidate = ytilde - zRoot
            max_candidate = ytilde + zRoot
            bounds = np.array([min_candidate, max_candidate]) 

        else:  # Handle array output
            len_y = len(ytilde)
            deltaF = np.zeros((n, len_y))
            min_candidate = np.empty(len_y)
            max_candidate = np.empty(len_y)

            for k in tqdm.tqdm(range(n), desc="Calculating Cauchy deviates"):
                r = np.random.rand(x.shape[0])
                c = np.tan(np.pi * (r - 0.5))
                K = np.max(c)
                delta = Delta * c / K
                x_sample = xtilde - delta
                result = f(x_sample)
                for i in range(len_y):
                    deltaF[k, i] = K * (ytilde[i] - result[i])

            for i in range(len_y):
                mask = np.isnan(deltaF[:, i])
                filtered_deltaF_i = deltaF[:, i][~mask]
                Z = lambda Del: n/2 - np.sum(1 / (1 + (filtered_deltaF_i / Del)**2))
                try:  # Handle potential errors in brentq
                    zRoot = brentq(Z, 1e-6, max(filtered_deltaF_i)/2)  # Use a small value for the lower bound
                except ValueError:
                    print(f"Warning: brentq failed for output {i}. Using 0 for zRoot.")
                    zRoot = 0  # Or handle the error in another way
                min_candidate[i] = ytilde[i] - zRoot
                max_candidate[i] = ytilde[i] + zRoot
           
            # Create a 2D array for bounds in the array case
            bounds = np.vstack([min_candidate, max_candidate]) 

        results = {
            'bounds': bounds,
            'min': {
                'f': min_candidate,
                'x': None
            },
            'max': {
                'f': max_candidate,
                'x': None
            },
            'raw_data': {
                'f': None,
                'x': None
            }
        }
        if save_raw_data == 'yes':
            print("Input-Output raw data are NOT available for the Cauchy method!")

    elif  save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes' 
        x_samples = np.zeros((n, x.shape[0]))
        K_values = np.zeros(n)
        for k in tqdm.tqdm(range(n), desc="Calculating Cauchy deviates"):
            r = np.random.rand(x.shape[0])
            c = np.tan(np.pi * (r - 0.5))
            K = np.max(c)
            delta = Delta * c / K
            x_samples[k] = xtilde - delta
            K_values[k] = K

        # results['min']['x'] = None
        # results['min']['f'] = None
        # results['max']['x'] = None
        # results['max']['f'] = None
        results['raw_data'] = {'x': x_samples, 'K': K_values}
    
    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")

    return results

# # Example usage with different parameters for minimization and maximization
# f = lambda x: x[0] + x[1] + x[2]  # Example function

# # Determine input parameters for function and method
# x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
# n=50
# # Call the method
# y = cauchydeviate_method(x_bounds,f=None, n=n, save_raw_data = 'yes')

# # print the results
# print("-" * 30)
# print("bounds:", y['bounds'])
    
# print("-" * 30)
# print("Minimum:")
# print("x:", y['min']['x'])
# print("f:", y['min']['f'])

# print("-" * 30)
# print("Maximum:")
# print("x:", y['max']['x'])
# print("f:", y['max']['f'])

# print("-" * 30)
# print("Raw data:")
# print("x:", y['raw_data']['x'])
 