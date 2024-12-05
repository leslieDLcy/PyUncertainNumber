import numpy as np
from scipy.stats import qmc
import tqdm
from typing import Callable

def sampling_method(x: np.ndarray, f: Callable, n: int, method='monte_carlo', 
                    endpoints=False, save_raw_data='no'):
    """
    Performs uncertainty propagation through sampling-based methods.

    args:
        x (np.ndarray): A 2D NumPy array where each row represents an input variable and 
                            the two columns define its lower and upper bounds (interval).
        f (Callable): A callable function that takes a 1D NumPy array of input values and returns the 
                        corresponding output(s). Can be None, in which case only samples are generated.
        n (int): The number of samples to generate for the chosen sampling method.
        method (str, optional): The sampling method to use. Choose from:
                                 - 'monte_carlo': Monte Carlo sampling (random sampling from uniform distributions)
                                 - 'lhs': Latin Hypercube sampling (stratified sampling for better space coverage)
                                Defaults to 'montecarlo'.
        endpoints (bool, optional): If True, include the interval endpoints in the sampling. 
                                    Defaults to False. (Implementation not provided)
        save_raw_data (str, optional): Whether to save raw data. Options: 'yes', 'no'. 
                                        Defaults to 'no'.
    
    signature:
        sampling_method(x:np.ndarray, f:Callable, n:int, method ='montecarlo', endpoints=False, save_raw_data = 'no') -> dict of np.ndarrays

    note:
        - The function assumes that the intervals in `x` represent uniform distributions.
        - The implementation for including endpoints (`endpoints=True`) is not provided and needs to be added based on specific requirements.
        - If the `f` function returns multiple outputs, the `all_output` array will be 2-dimensional y and x for all x samples.

    returns:
        dict: A dictionary containing the results:
            - 'min': A dictionary for lower bound results (if f in not None)
                - 'x': Input values that produced the miminum output value(s) (if f is not None).
                - 'f': Minimum output value(s) (if f is not None).
            - 'max':  A dictionary for upper bound results (if f in not None)
                - 'x': Input values that produced the maximum output value(s) (if f is not None).
                - 'f': Maximum output value(s) (if f is not None).
            - 'raw_data': A dictionary containing raw data (if save_raw_data is 'yes'):
                - 'x': All generated input samples.
                - 'f': Corresponding output values for each input sample.
    example:
        #Define input intervals
        x = np.array([[1, 2], [3, 4], [5, 6]])

        # Define the function
        f = lambda x: x[0] + x[1] + x[2]
        
        # Run sampling method with n = 5
        y = sampling_method(x, f, n=5, method='monte_carlo', endpoints=False, save_raw_data='no')

        # Print the results
        print("-" * 30)
        print("Minimum:")
        print("x:", y['min']['x'])
        print("f:", y['min']['f'])

        print("-" * 30)
        print("Maximum:")
        print("x:", y['max']['x'])
        print("f:", y['max']['f'])

        print("-" * 30)
        print("Raw data")
        print("x:",y['raw_data']['x'])
        print("type_x:",type(y['raw_data']['x']))
        print("f:", y['raw_data']['f'])
    """
    m = x.shape[0]
    lo = x[:, 0]
    hi = x[:, 1]

    if method == 'monte_carlo':
        X = lo + (hi - lo) * np.random.rand(n, m)
    elif method == 'lhs':
        sampler = qmc.LatinHypercube(m)
        X = lo + (hi - lo) * sampler.random(n)
    else:
        raise ValueError("Invalid sampling method. Choose 'montecarlo' or 'lhs'.")

    if endpoints:
        pass  # Add endpoints to the samples (implementation depends on your specific needs)

    #results = {'all_x': X}  # Initialize results with all_x
    results = {}  # Initialize an empty dictionary    

    if f is not None:  # Only evaluate if f is provided
        all_output = np.array([f(xi) for xi in tqdm.tqdm(X, desc="Evaluating samples")])

        if all_output.ndim == 1:  # If f returns a single output
            all_output = all_output.reshape(-1, 1)  # Reshape to a column vector

         # Create a dictionary to store the results
        results = {
            'min': {
                'f': np.min(all_output, axis=0),
                },
            'max': {
                'f': np.max(all_output, axis=0),
                },
            'raw_data': {
                'f': None, 
                'x': None}   
            }
        
        results['min']['x'] = []
        results['max']['x'] = []

        for i in range(all_output.shape[1]):  # Iterate over outputs
            min_indices = np.where(all_output[:, i] == results['min']['f'][i])[0]
            max_indices = np.where(all_output[:, i] == results['max']['f'][i])[0]
            
            # Convert to 2D arrays (if necessary) and append
            results['min']['x'].append(X[min_indices].reshape(-1, m))  # Reshape to (-1, m)
            results['max']['x'].append(X[max_indices].reshape(-1, m))

        # Concatenate the arrays in the lists into 2D arrays (if necessary)
        if len(results['min']['x']) > 1:
            results['min']['x'] = np.concatenate(results['min']['x'], axis=0)
        else:
            results['min']['x'] = results['min']['x'][0]  # Take the first (and only) array

        if len(results['max']['x']) > 1:
            results['max']['x'] = np.concatenate(results['max']['x'], axis=0)
        else:
            results['max']['x'] = results['max']['x'][0]  # Take the first (and only) array
        
        if save_raw_data == 'yes':
            results['raw_data'] = {'f': all_output, 'x': X}

    elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
        results['raw_data'] = {'f': None, 'x': X}
    
    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")

    return results
