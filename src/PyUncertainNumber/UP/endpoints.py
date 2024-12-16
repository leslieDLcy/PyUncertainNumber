import numpy as np
import tqdm
from typing import Callable
from PyUncertainNumber.UP.mixed_uncertainty.cartesian_product import cartesian

def endpoints_method(x:np.ndarray, f:Callable, results:dict = None, save_raw_data = 'no'):

    """ 
    args:
        - x: A 2D NumPy array where each row represents an input variable and 
          the two columns define its lower and upper bounds (interval).
        - f: A callable function that takes a 1D NumPy array of input values and 
          returns the corresponding output(s).
        - save_raw_data: Controls the amount of data returned.
          - 'no': Returns only the minimum and maximum output values along with the 
                  corresponding input values.
          - 'yes': Returns the above, plus the full arrays of unique input combinations 
                  (`all_input`) and their corresponding output values (`all_output`).


    signature:
        endpoints_method(x:np.ndarray, f:Callable, save_raw_data = 'no') -> dict

    note:
        - Performs uncertainty propagation using the Endpoints Method. 
        - The function assumes that the intervals in `x` represent uncertainties and aims to provide conservative
          bounds on the output uncertainty.
        - If the `f` function returns multiple outputs, the `bounds` array will be 2-dimensional.

    return:
        - dict: A dictionary containing the results:
          - 'bounds': An np.ndarray of the bounds for each output parameter (if f is not None). 
          - 'min': A dictionary for lower bound results (if f is not None):
            - 'x': Input values that produced the minimum output value(s).
            - 'f': Minimum output value(s).
          - 'max': A dictionary for upper bound results (if f is not None):
            - 'x': Input values that produced the maximum output value(s).
            - 'f': Maximum output value(s).
          - 'raw_data': A dictionary containing raw data (if `save_raw_data` is 'yes'):
            - 'x': All generated input samples.
            - 'f': Corresponding output values for each input sample.

    # Example usage with different parameters for minimization and maximization
    f = lambda x: x[0] + x[1] + x[2]  # Example function

    # Determine input parameters for function and method
    x_bounds = np.array([[1, 2], [3, 4], [5, 6]])

    # Call the method
    y = endpoints_method(x_bounds, f)

    print("-" * 30)
    print("bounds:", y['raw_data']['bounds'])
    print("-" * 30)
    print("Minimum:")
    print("x:", y['raw_data']['min']['x'])
    print("f:", y['raw_data']['min']['f'])

    print("-" * 30)
    print("Maximum:")
    print("x:", y['raw_data']['max']['x'])
    print("f:", y['raw_data']['max']['f'])

    print("-" * 30)
    print("Raw data:")
    print("x:", y['raw_data']['x'])
    print("f:", y['raw_data']['f']) 

    """
    # Create a sequence of values for each interval based on the number of divisions provided 
    # The divisions may be the same for all intervals or they can vary.
    m = x.shape[0]
    print(f"Total number of input combinations for the endpoint method: {2**m}") 
    
    # create an array with the unique combinations of all intervals 
    X = cartesian(*x) 

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
    
    # propagates the epistemic uncertainty through subinterval reconstitution   
    if f is not None:
        all_output = np.array([f(xi) for xi in tqdm.tqdm(X, desc="Evaluating samples")])

        if all_output.ndim == 1:
            all_output = all_output.reshape(-1, 1)

        results = { 'raw_data':{
                        'min': {
                            'f': np.min(all_output, axis=0)
                                },
                        'max': {
                            'f': np.max(all_output, axis=0)
                            },
                            }
                        }
        
   
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
            
            # Convert to 2D arrays and append
            results['raw_data']['min']['x'].extend(X[min_indices])  # Use extend here
            results['raw_data']['max']['x'].extend(X[max_indices])  # Use extend here

        # Concatenate arrays if necessary
        if len(results['raw_data']['min']['x']) > 1:
            results['raw_data']['min']['x'] = np.array(results['raw_data']['min']['x']) 
        else:
            results['raw_data']['min']['x'] = np.array([results['raw_data']['min']['x'][0]])

        if len(results['raw_data']['max']['x']) > 1:
            results['raw_data']['max']['x'] = np.array(results['raw_data']['max']['x'])
        else:
            results['raw_data']['max']['x'] = np.array([results['raw_data']['max']['x'][0]])

        #if save_raw_data == 'yes':
        results['raw_data']['f'] = all_output
        results['raw_data']['x'] = X

    elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
        results['raw_data'] = {'f': None, 'x': X}
    
    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")

    return results

# example

# # Example usage with different parameters for minimization and maximization
# f = lambda x: x[0] + x[1] + x[2]  # Example function

# # Determine input parameters for function and method
# x_bounds = np.array([[1, 2], [3, 4], [5, 6]])

# # Call the method
# y = endpoints_method(x_bounds, f)

# print("-" * 30)
# #print("bounds:", y['raw_data']['bounds'])
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