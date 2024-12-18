import numpy as np
from scipy.stats import qmc
import tqdm
from typing import Callable

def index_to_bool_(index:np.ndarray,dim=2):

    """ Converts a vector of indices to an array of boolean pairs for masking.

  Args:
    index: A NumPy array of integer indices representing selected elements or categories. 
           The values in `index` should be in the range [0, dim-1].
    dim: The number of categories or dimensions in the output boolean array. Defaults to 2  

  signature:
    index_to_bool_(index:np.ndarray,dim=2) -> tuple

   note:
        - the augument `index` is an np.ndaray of the index of intervals.
        - the argument `dim` will specify the function mapping of variables to be propagated. 
    
    #If dim > 2,  e.g. (2,0,1,0) the array of booleans is [(0,0,1),(1,0,0),(0,1,0),(1,0,0)].

  Returns:
    - A NumPy array of boolean pairs representing the mask.
  """
    
    index = np.asarray(index,dtype=int)
    return np.asarray([index==j for j in range(dim)],dtype=bool)

def sampling_method(x: np.ndarray, f: Callable, n_sam: int = 500, method='monte_carlo', results:dict = None,     
                    save_raw_data='no', endpoints=False ):
    """
    args:
        x (np.ndarray): A 2D NumPy array where each row represents an input variable and 
                            the two columns define its lower and upper bounds (interval).
        f (Callable): A callable function that takes a 1D NumPy array of input values and returns the 
                        corresponding output(s). Can be None, in which case only samples are generated.
        n_sam (int): The number of samples to generate for the chosen sampling method.
        method (str, optional): The sampling method to use. Choose from:
                                 - 'monte_carlo': Monte Carlo sampling (random sampling from uniform distributions)
                                 - 'latin_hypercube': Latin Hypercube sampling (stratified sampling for better space coverage)
                                Defaults to 'monte_carlo'.
        endpoints (bool, optional): If True, include the interval endpoints in the sampling. 
                                    Defaults to False. 
        save_raw_data (str, optional): Whether to save raw data. Options: 'yes', 'no'. 
                                        Defaults to 'no'.
    
    signature:
        sampling_method(x:np.ndarray, f:Callable, n_sam:int, method ='montecarlo', endpoints=False, results:dict = None, save_raw_data = 'no') -> dict of np.ndarrays

    note:
        - The function assumes that the na in `x` represent uniform distributions.
        - If the `f` function returns multiple outputs, the `all_output` array will be 2-dimensional y and x for all x samples.

    returns:
        dict: A dictionary containing the results:
            - 'bounds': An np.ndarray of the bounds for each output parameter (if f is not None). 
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
        
        # Run sampling method with n = 500
        y = sampling_method(x, f, n_sam=500, method='monte_carlo', endpoints=False, save_raw_data='no')

        # Print the results
        print("-" * 30)
        print("Minimum:")
        print("x:", y['raw_data']['min']['x'])
        print("f:", y['raw_data']['min']['f'])

        print("-" * 30)
        print("Maximum:")
        print("x:", y['raw_data']['max']['x'])
        print("f:", y['raw_data']['max']['f'])

        print("-" * 30)
        print("Raw data")
        print("x:",y['raw_data']['x'])
        print("type_x:",type(y['raw_data']['x']))
        print("f:", y['raw_data']['f'])
    """
    if results is None:
        results = {
             'un': None,
           
            'raw_data': {                
                'x': None,
                'f': None,
                'min': {
                        'x': None,
                        'f': None
                    },
                'max': {
                        'x': None,
                        'f': None,
                    },
                'bounds': None
                }
            }
    print(f"Total number of input combinations for the {method} method: {n_sam}")    
    
    m = x.shape[0]
    lo = x[:, 0]
    hi = x[:, 1]

    if method == 'monte_carlo':
        X = lo + (hi - lo) * np.random.rand(n_sam, m)
    elif method == 'latin_hypercube':
        sampler = qmc.LatinHypercube(m)
        X = lo + (hi - lo) * sampler.random(n_sam)
    else:
        raise ValueError("Invalid sampling method. Choose 'monte_carlo' or 'latin_hypercube'.")

    if endpoints:
        m=x.shape[0]
        Total = 2**m # Total number of endpoint combination for the give x input variables
        X_end = np.zeros((Total, m))  # Initialize array for endpoint combinations
        for j in range(Total):
            index = tuple([j//2**h-(j//2**(h+1))*2 for h in range(m)]) # tuple of 0s and 1s
            itb = index_to_bool_(index).T
            X_end[j, :] = x[itb]
        X = np.vstack([X, X_end])  # Combine generated samples with endpoint combinations     

    if f is not None:  # Only evaluate if f is provided
        all_output = np.array([f(xi) for xi in tqdm.tqdm(X, desc="Evaluating samples")])

        if all_output.ndim == 1:  # If f returns a single output
            all_output = all_output.reshape(-1, 1)  # Reshape to a column vector

         # Create a dictionary to store the results
        results['raw_data']['min']['f'] = np.min(all_output, axis=0)
        results['raw_data']['max']['f'] = np.max(all_output, axis=0)

        if all_output.shape[1] == 1:  # Single output
            results['raw_data']['bounds'] = np.array([results['raw_data']['min']['f'][0], results['raw_data']['max']['f'][0]])
        else:  # Multiple outputs
            bounds = np.empty((all_output.shape[1], 2))
            for i in range(all_output.shape[1]):
                bounds[i, :] = np.array([results['raw_data']['min']['f'][i], results['raw_data']['max']['f'][i]])
            results['raw_data']['bounds'] = bounds
        
        results['raw_data']['min']['x'] = []
        results['raw_data']['max']['x'] = []

        for i in range(all_output.shape[1]):  # Iterate over outputs
            min_indices = np.where(all_output[:, i] == results['raw_data']['min']['f'][i])[0]
            max_indices = np.where(all_output[:, i] == results['raw_data']['max']['f'][i])[0]
            
            # Convert to 2D arrays (if necessary) and append
            results['raw_data']['min']['x'].append(X[min_indices].reshape(-1, m))  # Reshape to (-1, m)
            results['raw_data']['max']['x'].append(X[max_indices].reshape(-1, m))

        # Concatenate the arrays in the lists into 2D arrays (if necessary)
        if len(results['raw_data']['min']['x']) > 1:
            results['raw_data']['min']['x'] = np.concatenate(results['raw_data']['min']['x'], axis=0)
        else:
            results['raw_data']['min']['x'] = results['raw_data']['min']['x'][0]  # Take the first (and only) array

        if len(results['raw_data']['max']['x']) > 1:
            results['raw_data']['max']['x'] = np.concatenate(results['raw_data']['max']['x'], axis=0)
        else:
            results['raw_data']['max']['x'] = results['raw_data']['max']['x'][0]  # Take the first (and only) array
        
        #if save_raw_data == 'yes':
        results['raw_data']['f'] = all_output
        results['raw_data']['x'] = X

    elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
        results['raw_data'] = {'f':None, 'x': X}
    
    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")

    return results

# # Example usage with different parameters for minimization and maximization
# f = lambda x: x[0] + x[1] + x[2]  # Example function

# # Determine input parameters for function and method
# x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
# n=20
# # Call the method
# y = sampling_method(x_bounds,f=f, n_sam=n, method  ='latin_hypercube' , endpoints= True, save_raw_data = 'yes')
# print(y)
# print("-" * 30)
# print("Minimum:")
# print("x:", y['raw_data']['min']['x'])
# print("f:", y['raw_data']['min']['f'])

# print("-" * 30)
# print("Maximum:")
# print("x:", y['raw_data']['max']['x'])
# print("f:", y['raw_data']['max']['f'])

# print("-" * 30)
# print("Raw data:")
# print("x:", y['raw_data']['x'])
# print("f:", y['raw_data']['f']) 