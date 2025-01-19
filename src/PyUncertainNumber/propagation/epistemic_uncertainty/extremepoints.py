import numpy as np
import tqdm
from typing import Callable
from pyuncertainnumber.propagation.epistemic_uncertainty.cartesian_product import cartesian
from pyuncertainnumber.propagation.epistemic_uncertainty.extreme_point_func import extreme_pointX
from pyuncertainnumber.propagation.utils import Propagation_results


def extremepoints_method(x: np.ndarray, f: Callable,
                         results: Propagation_results = None,
                         save_raw_data='no') -> Propagation_results:  # Specify return type
    """ 
        Performs uncertainty propagation using the Extreme Point Method for monotonic functions. 
        This method estimates the bounds of a function's output by evaluating it at specific combinations of extreme values
        (lower or upper bounds) of the input variables. It is efficient for monotonic functions but might not be accurate for non-monotonic functions. 
        If the `f` function returns multiple outputs, the `bounds` array will be 2-dimensional.

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
        extremepoints_method(x:np.ndarray, f:Callable, results:dict, save_raw_data = 'no') -> dict


    return:
        - A Propagation_results object containing the results.
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

    Example:
        # Example usage with different parameters for minimization and maximization
        >>> f = lambda x: x[0] + x[1] + x[2]  # Example function
        >>> # Determine input parameters for function and method
        >>> x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
        >>> # Call the method
        >>> y = extremepoint_method(x_bounds, f)
        >>> # print results
        >>> y.print()

    """
    if results is None:
        results = Propagation_results()  # Create an instance of Propagation_results

    # create an array with the unique combinations of all intervals
    X = cartesian(*x)

    d = X.shape[1]  # Number of dimensions
    inds = np.array([1] + [2**i + 1 for i in range(d)])  # Generate indices
    # Select rows based on indices (adjusting for 0-based indexing)
    Xeval = X[inds - 1]

    print(
        f"Total number of input combinations for the endpoints extreme points method: {d + 3}")

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

        if num_outputs == 1:
            lower_bound[0] = f(Xsign[2*i, :])
            upper_bound[0] = f(Xsign[2*i + 1, :])
        else:
            for i in range(num_outputs):
                lower_bound[i] = f(Xsign[2*i, :])[i]
                upper_bound[i] = f(Xsign[2*i + 1, :])[i]

        min_indices = np.zeros((d, num_outputs))
        max_indices = np.zeros((d, num_outputs))
        for i in range(num_outputs):  # Iterate over outputs
            min_indices[:, i] = Xsign[2*i, :]
            max_indices[:, i] = Xsign[2*i+1, :]

        # Convert to 2D arrays (if necessary) and append
        for i in range(num_outputs):
            results.raw_data['min'] = np.append(
                results.raw_data['min'], {'x': min_indices[:, i], 'f': lower_bound[i]})
            results.raw_data['max'] = np.append(
                results.raw_data['max'], {'x': max_indices[:, i], 'f': upper_bound[i]})
            results.raw_data['bounds'] = np.vstack([results.raw_data['bounds'], np.array(
                [lower_bound[i], upper_bound[i]])]) if results.raw_data['bounds'].size else np.array([lower_bound[i], upper_bound[i]])

        results.add_raw_data(sign_x=signX)
        results.raw_data['x'] = Xeval
        results.raw_data['f'] = all_output

    elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
        # Store Xeval in raw_data['x'] even if f is None
        results.add_raw_data(x=Xeval)

    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")

    return results

# # Example usage with different parameters for minimization and maximization
# f = lambda x: x[0] + x[1] + x[2]  # Example function

# # Determine input parameters for function and method
# x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
# n=2
# # Call the method
# y = extremepoints_method(x_bounds, f=None, save_raw_data= 'yes')
# y.print()
