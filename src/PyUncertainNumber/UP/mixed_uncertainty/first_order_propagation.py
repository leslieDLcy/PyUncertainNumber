import numpy as np
from typing import Callable
import tqdm
from PyUncertainNumber.UP.mixed_uncertainty.extremePointX import extreme_pointX
from PyUncertainNumber.UP.endpoints_monotonic import endpoints_monotonic_method
from PyUncertainNumber.UC.uncertainNumber import UncertainNumber as UN

def imp(X):
    """Imposition of intervals."""
    return np.array([np.max(X[0, :]), np.min(X[1, :])])

def first_order_propagation_method(x: list, f:Callable = None, results:dict = None,
                                   lim_Q: np.array = np.array([0.0001, 0.9999]), n_disc:int = 5):
    """
    args:
        x (list): A list of UncertainNumber objects.
        f (Callable): The function to evaluate.
        results (dict): A dictionary to store the results (optional).
        lim_Q (np.array): Quantile limits for discretization.
        n_disc (int): Number of discretization points.
    
    signature:
        first_order_propagation_method(x: list, f: Callable, results: dict, lim_Q: np.array, n_disc: int, save_raw_data='no') -> dict

    notes:
        Performs first-order uncertainty propagation for mixed uncertain numbers 
    
    returns:
        dict: A dictionary containing the results


    example:
        #TODO to add an example for this method.
    """
    if f is None:
        raise ValueError("This method requires the function to be known!")
    
    if results is None:
        results = {
             'un': None,
            'bounds': None, 
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

    d = len(x) # dimension of uncertain numbers 
    xl = []
    xr = []
    ranges = np.zeros((2,d))
    n_slices = np.zeros((d), dtype=int)
    u_list = []  # List to store 'u' for each uncertain number. 

    for i, un in enumerate(x):
        print(f"Processing variable {i + 1} with essence: {un.essence}")

        if un.essence == "distribution":
            # perform outward directed discretisation 
            n_slices[i] = n_disc
            nd = n_disc + 1  
            distribution_family = un.distribution_parameters[0]

            if distribution_family == 'triang':
                u = np.linspace(0, 1, nd)
            else:
                u = np.linspace(lim_Q[0], lim_Q[1], nd)

        elif un.essence == "pbox":
            n_slices[i] = n_disc
            distribution_family = un.distribution_parameters[0]
            if distribution_family == 'triang': 
                u = np.linspace(0, 1, n_disc)
            else:
                u = np.linspace(lim_Q[0], lim_Q[1], n_disc)
        else:  # un.essence == "interval"
            n_slices[i] = 1
            u = np.array([0.0,1.0])  # Fake values , not needed anywhere

        u_list.append(u)  # Add 'u' to the list

        # Generate discrete p-boxes   
        temp_xl = []  # Temporary list to hold xl values for the current variable
        temp_xr = []  # Temporary list to hold xr values for the current variable
        
        match un.essence:
            case "distribution":
                # Calculate xl and xr for distributions (adjust as needed)
                temp_xl = un.ppf(u_list[i][:-1]).tolist()  # Assuming un.ppf(u) returns a list or array
                temp_xr = un.ppf(u_list[i][1:]).tolist()  # Adjust based on your distribution
                ranges[:, i] = np.array([temp_xl[0], temp_xr[-1]])

            case "interval":
                temp_xl = np.array([un.bounds[0]]).tolist()  # Repeat lower bound for all quantiles
                temp_xr = np.array([un.bounds[1]]).tolist()  # Repeat upper bound for all quantiles
                ranges[:, i] = np.array([un.bounds])

            case "pbox":
                temp_xl = un.ppf(u_list[i])[0].tolist()  # Assuming un.ppf(u) returns a list of lists
                temp_xr = un.ppf(u_list[i])[1].tolist()
                ranges[:, i] = np.array([temp_xl[0], temp_xr[-1]])
            case _:
                raise ValueError(f"Unsupported uncertainty type: {un.essence}")

        xl.append(temp_xl)  # Add the temporary list to xl
        xr.append(temp_xr)  # Add the temporary list to xr
    
    # Determine the positive or negative signs for each input
    res = endpoints_monotonic_method(ranges.T, f)

    num_outputs = res['sign_x'].shape[0]
    inpsList = np.zeros((0, d))
    evalsList = np.zeros((0, num_outputs))

    all_output_list = []

    for out in range(num_outputs):
        for input in range(d):
            X = [ranges[:,k].tolist() for k in range(d)]
            temp_X = X.copy()
            temp_X[input] = []
            Xsings = np.empty((n_slices[input], 2, d))
            
            current_index = 0
            for slice in tqdm.tqdm(range(n_slices[input]), desc=f"Processing input {input+1}, output {out+1}"):
                temp_X[input] = []

                temp_X[input].extend(np.array([xl[input][slice], xr[input][slice]]).tolist())
                rang = np.array([temp_X[i] for i in range(d)], dtype=object)
                Xsings[slice, :, :] = extreme_pointX(rang, res['sign_x'][out])
                current_index += 1

                for k in range(Xsings.shape[1]):
                    c = Xsings[slice, k, :]
                    im = np.where((inpsList == c).all(axis=1))[0]
                    if not im.size:
                        output = f(c)
                        all_output_list.append(output)
                        inpsList = np.vstack([inpsList, c])
                        evalsList = np.vstack([evalsList, output])
                    else:
                        all_output_list.append(evalsList[im[0]])
            
            current_index = 0  # Reset for the next input
    
    # Convert all_output to a 2D NumPy array
    all_output = np.array(all_output_list, dtype=object).reshape(num_outputs, n_slices.sum(),2, -1)

     # Calculate min and max for each sublist in all_output
    min_values = np.min(all_output, axis=2) 
    max_values = np.max(all_output, axis=2) 

    lower_bound = np.zeros((num_outputs,min_values.shape[1]))
    upper_bound = np.zeros((num_outputs,max_values.shape[1]))
  
    bounds = np.empty((num_outputs, 2, lower_bound.shape[1])) 

    for i in range(num_outputs):
        min_values[:,i] = np.sort(min_values[:,i])
        max_values[:,i]  = np.sort(max_values[:,i])
        lower_bound[i,:] = min_values[:, i]  # Extract each column
        upper_bound[i,:] = max_values[:, i]  
                    
        bounds[i, 0, :] = lower_bound[i,:]
        bounds[i, 1, :] = upper_bound[i,:]   

        results['bounds']= bounds
        results['min']['f']=  lower_bound
        results['max']['f']=  upper_bound

    return results
