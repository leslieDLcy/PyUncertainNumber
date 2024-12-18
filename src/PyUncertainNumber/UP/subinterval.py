import numpy as np
import tqdm
from typing import Callable
from PyUncertainNumber.UP.mixed_uncertainty.cartesian_product import cartesian

def subinterval_method(x:np.ndarray, f:Callable, n_sub:np.array =3, results:dict = None,  save_raw_data = 'no'):

    """ 
    args:
        - x: A 2D NumPy array where each row represents an input variable and the two columns
           define its lower and upper bounds (interval).
        - f: A callable function that takes a 1D NumPy array of input values and returns the
           corresponding output(s).
        - n_sub: A scalar (integer) or a 1D NumPy array specifying the number of subintervals for 
           each input variable. 
            - If a scalar, all input variables are divided into the same number of subintervals (defaults 3 divisions).
            - If an array, each element specifies the number of subintervals for the 
              corresponding input variable.
        - save_raw_data: Controls the amount of data returned:
            - 'no': Returns only the minimum and maximum output values along with the 
                   corresponding input values.
            - 'yes': Returns the above, plus the full arrays of unique input combinations 
                     (`all_input`) and their corresponding output values (`all_output`).

    signature:
        subinterval_method(x:np.ndarray, f:Callable, n:np.array, results:dict = None, save_raw_data = 'no') -> dict

    note:
        - The function assumes that the intervals in `x` represent uncertainties and aims to provide conservative 
          bounds on the output uncertainty.
        - The computational cost increases exponentially with the number of input variables 
          and the number of subintervals per variable.
        - If the `f` function returns multiple outputs, the `all_output` array will be 2-dimensional.

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
    
    example:
        #Define input intervals
        x = np.array([[1, 2], [3, 4], [5, 6]])

        # Define the function
        f = lambda x: x[0] + x[1] + x[2]
        
        # Run sampling method with n = 2
        y = subinterval_method(x, f, n_sub, save_raw_data = 'yes')

        # Print the results
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
    
    # Create a sequence of values for each interval based on the number of divisions provided 
    # The divisions may be the same for all intervals or they can vary.
    m = x.shape[0]
    print(f"Total number of input combinations for the subinterval method: {(n_sub+1)**m}")
    
    if type(n_sub) == int: #All inputs have identical division
        total = (n_sub + 1)**m 
        Xint = np.zeros((0,n_sub + 1), dtype=object )
        for xi in x:
            new_X =  np.linspace(xi[0], xi[1], n_sub+1)
            Xint = np.concatenate((Xint, new_X.reshape(1,n_sub+1)),  axis=0)
    else: #Different divisions per input
        total = 1
        Xint = []   
        for xc, c in zip(x, range(len(n_sub))):
            total *= (n_sub[c]+1)
            new_X =  np.linspace(xc[0], xc[1], n_sub[c]+1)
            
            Xint.append(new_X)
 
        Xint = np.array(Xint, dtype=object) 
    # create an array with the unique combinations of all subintervals 
    X = cartesian(*Xint) 
    
    # propagates the epistemic uncertainty through subinterval reconstitution   
    if f is not None:
        all_output = np.array([f(xi) for xi in tqdm.tqdm(X, desc="Evaluating samples")])

        if all_output.ndim == 1:
            all_output = all_output.reshape(-1, 1)

        results = { 'raw_data':{
                        'min': {
                            'f': np.min(all_output, axis=0),
                            },
                        'max': {
                            'f': np.max(all_output, axis=0),
                            }

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
        results['raw_data'] = {'f': None, 'x': X}
    
    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")

    return results

# # Example usage with different parameters for minimization and maximization
# f = lambda x: x[0] + x[1] + x[2]  # Example function

# # Determine input parameters for function and method
# x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
# n=2
# # Call the method
# y = subinterval_method(x_bounds, f, n_sub=n, save_raw_data = 'yes')

# #Print the results
# print("-" * 30)
# print("bounds:", y['raw_data']['bounds'])

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