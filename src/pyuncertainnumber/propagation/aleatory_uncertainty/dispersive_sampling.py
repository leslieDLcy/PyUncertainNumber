import numpy as np
import tqdm
from typing import Callable, Union
from scipy.stats import qmc  # Import Latin Hypercube Sampling from SciPy
from pyuncertainnumber.propagation.utils import Propagation_results
from pyuncertainnumber.propagation.epistemic_uncertainty.cartesian_product import cartesian

def dispersive_sampling_method(
    x: list,
    f: Callable,
    results: Propagation_results = None,
    method="disperive_monte_carlo",
    n_sam: int = 500,
    tOp: Union[float, np.ndarray]= 0.999,
    bOt: Union[float, np.ndarray]=0.001,
    part_derv_sign: np.ndarray = None,
    save_raw_data="no") -> Propagation_results:  # Specify return type
    """Performs aleatory uncertainty propagation using a combination of Monte Carlo sampling and the extremepoints method.
       This function propagates aleatory uncertainty in the input variables by generating random samples from their distributions.
       It accounts for dependencies between input variables and multiple outputs by using the signs of partial derivatives 
       (`part_derv_sign`) to guide the correlation structure of the samples.
 
    args:
        x (list): A list of `UncertainNumber` objects, each representing an input
                 variable with its associated uncertainty.
        f (Callable): A callable function that takes a 1D NumPy array of input
                  values and returns the corresponding output(s). Can be None,
                  in which case only samples are generated.
        results (Propagation_results, optional): An object to store propagation results.
                                            Defaults to None, in which case a new
                                            `Propagation_results` object is created.
        method (str, optional): The sampling method to use. Choose from:
                            - 'dispersive_monte_carlo': Monte Carlo sampling accounting for 
                              perfectly positive or negative output-input based on partial derivatives output(s).
                            Defaults to 'dispersive_monte_carlo'.
        n_sam (int): The number of samples to generate for the chosen sampling method.
                Defaults to 500.
        save_raw_data (str, optional): Acts as a switch to enable or disable the storage of raw input data when a function (f) 
          is not provided.
          - 'no': Returns an error that no function is provided.
          - 'yes': Returns the full arrays of unique input combinations.

    signature:
        disperive_sampling_method(x: list, f: Callable, ...) -> propagation_results

    note: 
        The function evaluates the provided callable function (`f`) at the generated samples to estimate the 
            distribution of possible outputs.
        This fuction allows only for the propagation of independent variables.
        It can handle both single and multiple output functions

    returns:
        Returns `Propagation_results` object(s) containing:
            - 'un': UncertainNumber object(s) to represent the empirical distribution(s) of the output(s).
            - 'raw_data' (dict): Dictionary containing raw data shared across output(s):
                    - 'x' (np.ndarray): Input values.
                    - 'f' (np.ndarray): Output(s) values.

    raises:
        ValueError if no function is provided and save_raw_data is 'no' and if invald UP method is selected.

    example:
        # Example usage with different parameters for minimization and maximization
        >>> results = disperive_sampling_method(x=x, f=f, n_sam = 300, method = 'dispersive_monte_carlo', save_raw_data = "no")
    """
    if results is None:
        results = Propagation_results()

    if save_raw_data not in ("yes", "no"):
        raise ValueError("Invalid save_raw_data option. Choose 'yes' or 'no'.")
     
    match method:        
        case "dispersive_monte_carlo":
            d = len(x)  # dimension of uncertain numbers
            # Generate discrete p-boxes
            temp_xl = []  # Temporary list to hold xl values for the current variable
            temp_xr = []  # Temporary list to hold xr values for the current variable

            ranges = np.zeros((2, d))

            for i, un in enumerate(x):

                if un.essence == "distribution":

                    distribution_family = un.distribution_parameters[0]

                    if isinstance(tOp, np.ndarray):
                        top = tOp[i]
                    else:
                        top = tOp
                    if isinstance(bOt, np.ndarray):
                        bot = bOt[i]
                    else:
                        bot = bOt

                    if distribution_family == 'triang':
                        u = np.array([0.0, 1.0]) 
                    else:
                        u = np.array([bot, top])  # Use bOt and tOp here
                    
                    temp_xl = un.ppf(u[0]).tolist()
                    # Adjust based on your distribution
                    temp_xr = un.ppf(u[1]).tolist()
                    ranges[:, i] = np.array([temp_xl, temp_xr])

                else:  # un.essence == "interval"
                    raise ValueError("Only inputs expressed as distributions are supported")
        
            # create an array with the unique combinations of all intervals
            X = cartesian(*ranges.T)

            d = X.shape[1]  # Number of dimensions
            inds = np.array([1] + [2**i + 1 for i in range(d)])  # Generate indices
            # Select rows based on indices (adjusting for 0-based indexing)
            Xeval = X[inds - 1]

            # propagates the epistemic uncertainty through the extremepoints method
            if part_derv_sign is None and f is not None:
                # Simulate function for the selected subset
                all_output = []
                for c in tqdm.tqdm(Xeval, desc="Number of function evaluations"):
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
                part_deriv_sign = np.zeros((num_outputs, d))

                for i in range(num_outputs):
                    # Calculate signs based on initial output values
                    part_deriv_sign[i] = np.sign(all_output[1:, i] - all_output[0, i])[::-1]
                
                num_outputs = len(part_deriv_sign)
                base_samples = np.round(np.random.rand(n_sam), decimals=6)
                parameter_samples = np.zeros((n_sam, d, num_outputs))  # Pre-allocate with an extra dimension for outputs
                
                for k in range(num_outputs):
                    for i, un in enumerate(x):
                        if part_deriv_sign[k, i] == 1:  # Positive correlation
                            parameter_samples[:, i, k] = un.ppf(base_samples[:])
                        elif part_deriv_sign[k, i] == -1:  # Negative correlation
                            parameter_samples[:, i, k] = un.ppf(1 - base_samples[:])
                        else:  # Independent (no correlation)
                            parameter_samples[:, i, k] = un.random(size=n_sam)

                # Optimized function evaluation
                evaluated_points = {}  # Store evaluated points
                all_output = np.zeros((n_sam, num_outputs))  # Pre-allocate for all outputs
                num_evaluations = 0 

                for i in tqdm.tqdm(range(n_sam), desc="Input samples"):
                    for k in range(num_outputs):
                        input_key = tuple(np.round(parameter_samples[i,:, k], decimals = 5))
                        if input_key not in evaluated_points:
                            output = f(parameter_samples[i,:, k])
                            evaluated_points[input_key] = output
                            num_evaluations += 1
                        else:
                            output = evaluated_points[input_key]
                        all_output[i, k] = output[k]  # Store the output for the current sample and output
                
                print(f"Number of total function evaluations: {num_evaluations + len(Xeval)}") 
                
                if all_output.ndim == 1:  # If f returns a single output
                    # Reshape to a column vector
                    all_output = all_output.reshape(-1, 1)      
            
                #elif save_raw_data == "yes":  # If f is None and save_raw_data is 'yes'
                results.add_raw_data(x=parameter_samples)
                results.add_raw_data(f=all_output)

            elif part_derv_sign is None and f is None:
                raise ValueError("Provide the partial defivative signs (part_derv_sign)")
            else:
                raise ValueError("No calculation is possible!")
            
        case _: raise ValueError(
                     "Invalid UP method!")
    return results