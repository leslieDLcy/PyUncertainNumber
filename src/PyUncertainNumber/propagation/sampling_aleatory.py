import numpy as np
import tqdm
from typing import Callable
from scipy.stats import qmc  # Import Latin Hypercube Sampling from SciPy
from PyUncertainNumber.UP.utils import propagation_results

def sampling_aleatory_method(x: list, f: Callable, 
                             results: propagation_results = None,
                             n_sam: int = 500, method='monte_carlo',      
                             save_raw_data='no')-> propagation_results:  # Specify return type
    """
    args:
        x (list): A list of UncertainNumber objects, each representing an input 
                  variable with its associated uncertainty.
        f (Callable): A callable function that takes a 1D NumPy array of input 
                      values and returns the corresponding output(s). Can be None, 
                      in which case only samples are generated.
        n_sam (int): The number of samples to generate for the chosen sampling method (default 500 samples).
        method (str, optional): The sampling method to use. Choose from:
                                - 'monte_carlo': Monte Carlo sampling (random sampling 
                                                 from the distributions specified in 
                                                 the UncertainNumber objects)
                                - 'latin_hypercube': Latin Hypercube sampling (stratified 
                                                      sampling for better space coverage)
                                Defaults to 'monte_carlo'.
        save_raw_data (str, optional): Whether to save raw data. Options: 'yes', 'no'. 
                                    Defaults to 'no'.
    
    signature:
        sampling_aleatory_method(x:list, f:Callable, n_sam:int, method ='montecarlo', results:dict = None,    
                                    save_raw_data = 'no') -> dict of np.ndarrays

    note:
        - Performs uncertainty propagation using Monte Carlo or Latin Hypercube sampling, 
          similar to the `sampling_method`. Calculates Kolmogorov-Smirnov (K-S) bounds 
          for the outputs to provide a non-parametric confidence region for the uncertainty in the results.
        - If the `f` function returns multiple outputs, the `all_output` array will be 2-dimensional y and x for all x samples.

    returns:
        dict: A dictionary containing the results:
            - 'ks_bounds': A list of dictionaries, one for each output of f, containing:
                - 'x_vals': Values at which the ECDF and K-S bounds are evaluated.
                - 'upper_bound': Upper K-S bound.
                - 'lower_bound': Lower K-S bound.
            - 'raw_data': A dictionary containing raw data (if save_raw_data is 'yes'):
                - 'x': All generated input samples.
                - 'f': Corresponding output values for each input sample.
    example:

    """
    if results is None:
        results = propagation_results()
    
    if method not in ('monte_carlo', 'latin_hypercube'):
        raise ValueError("Invalid sampling method. Choose 'monte_carlo' or 'latin_hypercube'.")

    if save_raw_data not in ('yes', 'no'):
        raise ValueError("Invalid save_raw_data option. Choose 'yes' or 'no'.")
    
    print(f"Total number of input combinations for the {method} method: {n_sam}")
    
    if results is None:
        results = {
                'un': None, 
            
                'raw_data': {
                        'f': None,
                        'x': None
                        }
                }
    
    if method == 'monte_carlo':   
        parameter_samples = np.array([
            un.random(size=n_sam) for un in x
        ])

    elif method == 'latin_hypercube':
        sampler = qmc.LatinHypercube(d=len(x))
        lhd_samples = sampler.random(n=n_sam)

        parameter_samples = []  # Initialize an empty list to store the samples
 
        for i, un in enumerate(x):  # Iterate over each UncertainNumber in the list 'x'
            q_values = lhd_samples[:, i]  # Get the entire column of quantiles for this UncertainNumber
    
            # Now we need to calculate the ppf for each q value in the q_values array
            ppf_values = []  # Initialize an empty list to store the ppf values for this UncertainNumber
            for q in q_values:  # Iterate over each individual q value
                ppf_value = un.ppf(q)  # Calculate the ppf value for this q
                ppf_values.append(ppf_value)  # Add the calculated ppf value to the list

            parameter_samples.append(ppf_values)  # Add the list of ppf values to the main list

        parameter_samples = np.array(parameter_samples)  # Convert the list of lists to a NumPy array
        
    # Transpose to have each row as a sample
    parameter_samples = parameter_samples.T
     
    if f is not None:  # Only evaluate if f is provided
        all_output = np.array([f(xi) for xi in tqdm.tqdm(parameter_samples, desc="Evaluating samples")])

        if all_output.ndim == 1:  # If f returns a single output
            all_output = all_output.reshape(-1, 1)  # Reshape to a column vector
            
        #if save_raw_data == 'yes':
        results.add_raw_data(x = parameter_samples)
        results.add_raw_data(f = all_output)

    elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
        results.add_raw_data(x = parameter_samples)
    
    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")

    return results

# # example
# from PyUncertainNumber import UncertainNumber as UN

# def cantilever_beam_deflection(x):
#     """Calculates deflection and stress for a cantilever beam.

#     Args:

#         x (np.array): Array of input parameters:
#             x[0]: Length of the beam (m)
#             x[1]: Second moment of area (mm^4)
#             x[2]: Applied force (N)
#             x[3]: Young's modulus (MPa)

#     Returns:
#         float: deflection (m)
#                Returns np.nan if calculation error occurs.
#     """

#     beam_length = x[0]
#     I = x[1]
#     F = x[2]
#     E = x[3]
#     try:  # try is used to account for cases where the input combinations leads to error in fun due to bugs
#         deflection = F * beam_length**3 / (3 * E * 10**6 * I)  # deflection in m
        
#     except:
#         deflection = np.nan

#     return deflection
# # example
# L = UN(name='beam length', symbol='L', units='m', essence='distribution', distribution_parameters=["gaussian", [10.05, 0.033]])
# I = UN(name='moment of inertia', symbol='I', units='m', essence='distribution', distribution_parameters=["gaussian", [0.000454, 4.5061e-5]])
# F = UN(name='vertical force', symbol='F', units='kN', essence='distribution', distribution_parameters=["gaussian", [24, 8.67]])
# E = UN(name='elastic modulus', symbol='E', units='GPa', essence='distribution', distribution_parameters=["gaussian", [210, 6.67]])

# METHOD = "monte_carlo"
# base_path = ""

# a = sampling_aleatory_method(x=[L, I, F, E], #['L', 'I', 'F', 'E'], 
#           f = None,#cantilever_beam_deflection, 
#           n_sam = 3, 
#           method = METHOD, 
#           save_raw_data = "yes"
#          )

# a.print()

