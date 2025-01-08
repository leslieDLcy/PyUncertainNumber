import numpy as np
from typing import Callable, Union
import tqdm
from PyUncertainNumber.UP.extreme_point_func import extreme_pointX
from PyUncertainNumber.UP.extremepoints import extremepoints_method
from PyUncertainNumber.UP.utils import propagation_results, condense_bounds

def imp(X):
    """Imposition of intervals."""
  
    return np.array([np.max(X[:, 0]), np.min(X[:, 1])])

def first_order_propagation_method(x: list, f:Callable = None,  
                                results: propagation_results = None, 
                                #method = 'extremepoints',
                                n_disc: Union[int, np.ndarray] = 10, 
                                condensation: Union[float, np.ndarray] = None, 
                                tOp: Union[float, np.ndarray] = 0.999,
                                bOt: Union[float, np.ndarray] = 0.001,
                                save_raw_data= 'no')-> propagation_results:  # Specify return type

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
    d = len(x) # dimension of uncertain numbers 
    results = propagation_results()
    xl = []
    xr = []
    ranges = np.zeros((2,d))
    n_slices = np.zeros((d), dtype=int)
    u_list = []  # List to store 'u' for each uncertain number. 

    for i, un in enumerate(x):
        print(f"Processing variable {i + 1} with essence: {un.essence}")

        if un.essence == "distribution":

            distribution_family = un.distribution_parameters[0]
            
            # perform outward directed discretisation
            if isinstance(n_disc, int):
                nd = n_disc + 1
            else:
                nd = n_disc[i] + 1  # Use the i-th element of n_disc array 
            n_slices[i] = nd
            
            if isinstance(tOp, np.ndarray):
                top = tOp[i]
            else:
                top = tOp
            if isinstance(bOt, np.ndarray):
                bot = bOt[i]
            else:
                bot = bOt

            if distribution_family == 'triang':
                u = np.linspace(0, 1, nd)
            else:             
                u = np.linspace(bot, top, nd)  # Use bOt and tOp here

        elif un.essence == "pbox":
            distribution_family = un.distribution_parameters[0]
            if isinstance(n_disc, int):
                    nd = n_disc
            else:
                    nd = n_disc[i]  # Use the i-th element of n_disc array            
            n_slices[i] = nd ## this is wrong, why do we even need it?

            if isinstance(tOp, np.ndarray):
                top = tOp[i]
            else:
                top = tOp
            if isinstance(bOt, np.ndarray):
                bot = bOt[i]
            else:
                bot = bOt
            
            if distribution_family == 'triang':                 
                u = np.linspace(0, 1, nd)
            else:
                u = np.linspace(bot, top, nd)  # Use bOt and tOp here
                
        else:  # un.essence == "interval"
            u = np.array([0.0, 1.0])  # Or adjust as needed for intervals
            n_slices[i] = 2

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
    res = extremepoints_method(ranges.T, f)

    # Determine the number of outputs from the first evaluation
    try:
        num_outputs = res.raw_data['sign_x'].shape[0]
    except TypeError:
        num_outputs = 1  # If f returns a single value

    all_output_list = []
    for _ in range(num_outputs):
        output_for_current_output = []
        for input in range(d):
            slices_for_input = []
            for _ in range(n_slices[input]-1):  # Use n_slices[input] here
                slices_for_input.append([None, None])
            output_for_current_output.append(slices_for_input)
        all_output_list.append(output_for_current_output)

    inpsList = np.zeros((0, d))
    evalsList = np.zeros((0, num_outputs))

    inpsList = np.zeros((0, d))
    evalsList = np.zeros((0, num_outputs))

    all_output_list = []
    for _ in range(num_outputs):
        output_for_current_output = []
        for input in range(d):
            slices_for_input = []
            for _ in range(n_slices[input] - 1):
                slices_for_input.append([None, None])
            output_for_current_output.append(slices_for_input)
        all_output_list.append(output_for_current_output)

    inpsList = np.zeros((0, d))
    evalsList = np.zeros((0, num_outputs))  # Store results for all outputs

    for input in range(d):  # Iterate over input variables first
        X = [ranges[:, k].tolist() for k in range(d)]
        temp_X = X.copy()
        Xsings = np.empty((n_slices[input], 2, d))
        current_index = 0

        for slice in tqdm.tqdm(range(n_slices[input] - 1), desc=f"Processing input {input+1}"):
            temp_X[input] = []
            temp_X[input].extend(np.array([xl[input][slice], xr[input][slice]]).tolist())
            rang = np.array([temp_X[i] for i in range(d)], dtype=object)
            Xsings[slice, :, :] = extreme_pointX(rang, res.raw_data['sign_x'])  # Use the entire sign_x array
            current_index += 1

            for k in range(Xsings.shape[1]):
                c = Xsings[slice, k, :]
                im = np.where((inpsList == c).all(axis=1))[0]
                if not im.size:
                    output = f(c)

                    # Store each output in a separate sublist
                    for out in range(num_outputs):
                        all_output_list[out][input][slice][k] = output[out]

                    inpsList = np.vstack([inpsList, c])
                    evalsList = np.vstack([evalsList, np.array(output)])
                else:
                    for out in range(num_outputs):
                        all_output_list[out][input][slice][k] = evalsList[im[0]][out]

            current_index = 0  # Reset for the next input

    # Reshape all_output based on the actual number of elements per output
    all_output = np.array(all_output_list, dtype=object)
   
    all_output = np.reshape(all_output, (num_outputs, d, -1, 2))  # Reshape to 4D
     
    # Calculate min and max for each output and input variable
    min_values = np.min(all_output, axis=3)
    max_values = np.max(all_output, axis=3)

    bounds_input = np.empty((num_outputs, d, n_disc,2))  # Initialize bounds_input

    for i in range(num_outputs):
        for j in range(d):  # Iterate over input variables
            # Merge min_values and max_values into bounds_input
            for k in range(n_disc):
                bounds_input[i, j, k, 0] = min_values[i, j, k]
                bounds_input[i, j, k, 1] = max_values[i, j, k]

            # Sort bounds_input along the last axis (k)
            bounds_input[i, j, :, :] = np.sort(bounds_input[i, j, :, :], axis=-1)

    lower_bound = np.zeros((num_outputs, n_disc))
    upper_bound = np.zeros((num_outputs, n_disc))

    bounds = np.empty((num_outputs, 2, n_disc))  # Initialize bounds
    lower_bound = np.zeros((num_outputs, n_disc))  # Initialize lower_bound
    upper_bound = np.zeros((num_outputs, n_disc))  # Initialize upper_bound

    for i in range(num_outputs):
        temp_bounds = []  # Temporary list for bounds
 
        for k in range(n_disc):  # Iterate over input variables
            # Impose the p-boxes for each input variable and store in temporary list
            temp_bounds.append(imp(bounds_input[i, :, k, :]))

        # Impose the p-boxes across all input variables for the current output
        bounds[i, :, :] = np.array(temp_bounds).T 

        # Extract lower_bound and upper_bound from bounds
        lower_bound[i, :] = bounds[i, 1, :]
        upper_bound[i, :] = bounds[i, 0, :]
   
    if condensation is not None:
        bounds = condense_bounds(bounds, condensation)

    results.raw_data['bounds'] = bounds
    results.raw_data['min'] = np.array([{ 'f': lower_bound}])
    results.raw_data['max'] = np.array([{ 'f': upper_bound}])

    return results

from PyUncertainNumber import UncertainNumber

def myFunctionWithTwoOutputs(x):
    """
    Example function with two outputs.
    Replace this with your actual function logic.
    """
    input1= x[0]
    input2=x[1]
    input3=x[2]
    input4=x[3]
    input5=x[4]
    output1 = input1 + input2 + input3 + input4 + input5
    output2 = input1 * input2 * input3 * input4 * input5
    return output1, output2

means = np.array([1, 2, 3, 4, 5])
stds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
x = np.array([1, 2, 3, 4, 5])
a = myFunctionWithTwoOutputs(x)
print(a[0])

n_disc = 10  # Number of discretizations

x = [
        UncertainNumber(essence='distribution', distribution_parameters=["gaussian",[means[0], stds[0]]]),
        UncertainNumber(essence = 'distribution', distribution_parameters= ["gaussian",[means[1], stds[1]]]),
        UncertainNumber(essence = 'distribution', distribution_parameters= ["gaussian",[means[2], stds[2]]]),
        UncertainNumber(essence = 'distribution', distribution_parameters= ["gaussian",[means[3], stds[3]]]),
        UncertainNumber(essence = 'distribution', distribution_parameters= ["gaussian",[means[4], stds[4]]]),
    ]

results = first_order_propagation_method(x=x, f=myFunctionWithTwoOutputs, n_disc=n_disc)

print(results)


