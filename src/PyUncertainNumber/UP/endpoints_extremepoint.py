import numpy as np
import tqdm
from typing import Callable
from PyUncertainNumber.UP.mixed_uncertainty.cartesian_product import cartesian
from PyUncertainNumber.UP.mixed_uncertainty.extremePointX import extreme_pointX

def endpoints_monotonic_method(x:np.ndarray, f:Callable, save_raw_data = 'no'):

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
        endpoints_monotonic_method(x:np.ndarray, f:Callable, save_raw_data = 'no') -> dict

    note:
        - Performs uncertainty propagation using the Extreme Point Method for monotonic functions. 
        - This method estimates the bounds of a function's output by evaluating it at specific
          combinations of extreme values (lower or upper bounds) of the input variables. 
        - It is efficient for monotonic functions but might not be accurate for non-monotonic functions.
        - If the `f` function returns multiple outputs, the `bounds` array will be 2-dimensional.

    return:
        - dict: A dictionary containing the results:
          - 'bounds': An np.ndarray of the bounds for each output parameter (if f is not None). 
          - 'sign_x': A NumPy array of shape (num_outputs, d) containing the signs (i.e., positive, negative) 
                    used to determine the extreme points for each output.
          - 'min': A dictionary for lower bound results (if f is not None):
            - 'x': Input values that produced the minimum output value(s).
            - 'f': Minimum output value(s).
          - 'max': A dictionary for upper bound results (if f is not None):
            - 'x': Input values that produced the maximum output value(s).
            - 'f': Maximum output value(s).
          - 'raw_data': A dictionary containing raw data (if `save_raw_data` is 'yes'):
            - 'x': All generated input samples.
            - 'f': Corresponding output values for each input sample.

    example:
        #Define input intervals
        x = np.array([[1, 2], [3, 4], [5, 6]])

        # Define the function
        f = lambda x: x[0] + x[1] + x[2]
        
        # Run sampling method with n = 2
        y = endpoints_method(x, f, save_raw_data = 'yes')
        
        # Print the results
        print("-" * 30)
        print("bounds:", y['bounds'])

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
    # create an array with the unique combinations of all intervals 
    X = cartesian(*x) 

    d = X.shape[1]  # Number of dimensions
    inds = np.array([1] + [2**i + 1 for i in range(d)])  # Generate indices
    Xeval = X[inds - 1]  # Select rows based on indices (adjusting for 0-based indexing)

    print(f"Total number of input combinations for the endpoint sign method: {len(inds - 1)}") 

    results = {
        'un': None,
        'bounds': None, 
        'signs_x': None,
        'min': {
                'x': None,
                'f': None
            },
        'max': {
                'x': None,
                'f': None,
                },
        'raw_data': {
                'f': None,
                'x': None
                }
        }
    
    # propagates the epistemic uncertainty through subinterval reconstitution   
    if f is not None:

        # Simulate function for the selected subset
        all_output = []  
        for c in tqdm.tqdm(Xeval, desc="Evaluating samples"):
            output = f(c) 
            all_output.append(output)
        
        # Determine the number of outputs from the first evaluation
        try:
            num_outputs = len(all_output[0])
        except TypeError:
            num_outputs = 1  # If f returns a single value

        # Convert all_output to a NumPy array with the correct shape
        all_output = np.array(all_output).reshape(-1, num_outputs)  

        # Calculate signs
        signX = np.zeros((num_outputs, d))
        Xsign = np.zeros((2 * num_outputs, d))
        for i in range(num_outputs):
            # Calculate signs based on initial output values
            signX[i] = np.sign(all_output[1:, i] - all_output[0, i])[::-1]  

            # Calculate extreme points
            Xsign[2*i:2*(i+1), :] = extreme_pointX(x, signX[i])
        
        lower_bound = np.zeros(num_outputs)
        upper_bound = np.zeros(num_outputs)
        for i in range(num_outputs):
            lower_bound[i] = f(Xsign[2*i, :])
            upper_bound[i] = f(Xsign[2*i + 1, :])

        results = {
            'sign_x': signX,
            'min': {
                'f': lower_bound,
            },
            'max': {
                'f': upper_bound,
            }
        }
        if num_outputs == 1:  # Single output
            results['bounds'] = np.array([results['min']['f'][0], results['max']['f'][0]])
        else:  # Multiple outputs
            bounds = []
            for i in range(num_outputs):
                bounds.append([results['min']['f'][i], results['max']['f'][i]])
            results['bounds'] = np.array(bounds)
        
        min_indices = np.zeros((d,num_outputs))
        max_indices = np.zeros((d,num_outputs))
        for i in range(num_outputs):  # Iterate over outputs
            min_indices[:,i] = Xsign[2*i, :]
            max_indices[:,i] = Xsign[2*i+1, :]
            
        # Convert to 2D arrays (if necessary) and append
        results['min']['x'] = min_indices
        results['max']['x'] = max_indices

    elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
        results['raw_data'] = {'f': None, 'x': Xeval}
    
    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")

    return results

# # Example usage with different parameters for minimization and maximization
# f = lambda x: x[0] + x[1] + x[2]  # Example function

# # Determine input parameters for function and method
# x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
# n=2
# # Call the method
# y = endpoints_monotonic_method(x_bounds, f)
# print(y['sign_x'])
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
# print("f:", y['raw_data']['f']) 