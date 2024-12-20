import numpy as np
from typing import Callable, Union
import tqdm
from PyUncertainNumber.UP.cartesian_product import cartesian
from PyUncertainNumber.UP.extreme_point_func import extreme_pointX
from PyUncertainNumber.UP.extremepoints import extremepoints_method
from PyUncertainNumber.UP.utils import propagation_results

#TODO add tail concentratig algorithms and condensation.
#TODO add x valus for min and max
def second_order_propagation_method(x: list, f:Callable = None,  
                                    results: propagation_results = None, 
                                    method = 'endpoints',
                                    n_disc: Union[int, np.ndarray] = 10, 
                                    condensation:int = 1, # to be implemented
                                    tOp: Union[float, np.ndarray] = 0.999,
                                    bOt: Union[float, np.ndarray] = 0.001,
                                    save_raw_data= 'no')-> propagation_results:  # Specify return type
    """
    args:
        x (list): A list of UncertainNumber objects.
        f (Callable): The function to evaluate.
        results (dict): A dictionary to store the results (optional).
        method (str): The method which will estimate bounds of each combination of focal elements (default is the endpoint)
        lim_Q (np.array): Quantile limits for discretization.
        n_disc (int): Number of discretization points.
    
    signature:
        second_order_propagation_method(x: list, f: Callable, results: dict, method: str, lim_Q: np.array, n_disc: int) -> dict

    notes:
        Performs second-order uncertainty propagation for mixed uncertain numbers 
    
    returns:
        dict: A dictionary containing the results
    """
    d = len(x) # dimension of uncertain numbers 
    results = propagation_results()
    bounds_x = []  
    ranges = np.zeros((2,d))
    u_list = []  # List to store 'u' for each uncertain number
    
    for i, un in enumerate(x):
        print(f"Processing variable {i + 1} with essence: {un.essence}")

        if un.essence == "distribution":
            # perform outward directed discretisation 
            if isinstance(n_disc, int):
                nd = n_disc + 1
            else:
                nd = n_disc[i] + 1  # Use the i-th element of n_disc array

            distribution_family = un.distribution_parameters[0]

            if distribution_family == 'triang':
                u = np.linspace(0, 1, nd)
            else:
                if isinstance(tOp, np.ndarray):
                    top = tOp[i]
                else:
                    top = tOp
                if isinstance(bOt, np.ndarray):
                    bot = bOt[i]
                else:
                    bot = bOt
                u = np.linspace(bot, top, nd)  # Use bOt and tOp here


        elif un.essence == "pbox":
            distribution_family = un.distribution_parameters[0]
            if distribution_family == 'triang': 
                if isinstance(n_disc, int):
                    nd = n_disc
                else:
                    nd = n_disc[i]  # Use the i-th element of n_disc array
                u = np.linspace(0, 1, nd)
            else:
                if isinstance(n_disc, int):
                    nd = n_disc
                else:
                    nd = n_disc[i]  # Use the i-th element of n_disc array

                if isinstance(tOp, np.ndarray):
                    top = tOp[i]
                else:
                    top = tOp
                if isinstance(bOt, np.ndarray):
                    bot = bOt[i]
                else:
                    bot = bOt
                u = np.linspace(bot, top, nd)  # Use bOt and tOp here
                
        else:  # un.essence == "interval"
            u = np.array([0.0, 1.0])  # Or adjust as needed for intervals

        u_list.append(u)  # Add 'u' to the list

        # Generate discrete p-boxes
        match un.essence:
            case "distribution":
                # Calculate xl and xr for distributions (adjust as needed)
                temp_xl = un.ppf(u_list[i][:-1])  # Assuming un.ppf(u) returns a list or array
                temp_xr = un.ppf(u_list[i][1:])  # Adjust based on your distribution                
                rang = np.array([temp_xl, temp_xr]).T  # Create a 2D array of bounds
                bounds_x.append(rang)
                ranges[:, i] = np.array([temp_xl[0], temp_xr[-1]])

            case "interval":
                temp_xl = np.array([un.bounds[0]])  # Repeat lower bound for all quantiles
                temp_xr = np.array([un.bounds[1]])  # Repeat upper bound for all quantiles               
                rang = np.array([temp_xl, temp_xr]).T  # Create a 2D array of bounds
                bounds_x.append(rang)
                ranges[:, i] = un.bounds #np.array([un.bounds])

            case "pbox":
                temp_xl = un.ppf(u_list[i])[0]  
                temp_xr = un.ppf(u_list[i])[1]               
                rang = np.array([temp_xl, temp_xr]).T  # Create a 2D array of bounds
                bounds_x.append(rang)
                ranges[:, i] = np.array([temp_xl[0], temp_xr[-1]])

            case _:
                raise ValueError(f"Unsupported uncertainty type: {un.essence}")

    # Automatically generate merged_array_index
    bounds_x_index = [np.arange(len(sub_array)) for sub_array in bounds_x]
        
    # Calculate Cartesian product of indices using your cartesian function
    cartesian_product_indices = cartesian(*bounds_x_index)
    
    # Generate the final array using the indices
    focal_elements_comb = []
    for indices in cartesian_product_indices:
        temp = []
        for i, index in enumerate(indices):
            temp.append(bounds_x[i][index])
        focal_elements_comb.append(temp)

    focal_elements_comb = np.array(focal_elements_comb, dtype=object)
    all_output = None

    if f is not None:
        # Efficiency upgrade: store repeated evaluations
        inpsList = np.zeros((0, d))
        evalsList = []  
        numRun = 0

        match method:
            case "endpoints":
                x_combinations = np.empty(( focal_elements_comb.shape[0]*(2**d), d), dtype=float)  # Pre-allocate the array
                current_index = 0  # Keep track of the current insertion index

                for array in focal_elements_comb:
                    cartesian_product_x = cartesian(*array)
                    num_combinations = cartesian_product_x.shape[0]  # Get the number of combinations from cartesian(*array)
                    # Assign the cartesian product to the appropriate slice of x_combinations
                    x_combinations[current_index : current_index + num_combinations] = cartesian_product_x      
                    current_index += num_combinations  # Update the insertion index

                # Initialize all_output as a list to store outputs initially
                all_output_list = []
                evalsList = []
                numRun = 0
                inpsList = np.empty((0, x_combinations.shape[1]))  # Initialize inpsList with the correct number of columns

                for case in tqdm.tqdm(x_combinations, desc="Evaluating combinations"):  # Wrap the loop with tqdmx_combinations
                    im = np.where((inpsList == case).all(axis=1))[0]
                    if not im.size:
                        output = f(case)
                        all_output_list.append(output)
                        inpsList = np.vstack([inpsList, case])
                        evalsList.append(output)
                        numRun += 1
                    else:
                        all_output_list.append(evalsList[im[0]])

                # Determine num_outputs AFTER running the function
                try:
                    num_outputs = len(all_output_list[0])
                except TypeError:
                    num_outputs = 1

                # Convert all_output to a 2D NumPy array
                all_output =  np.array(all_output_list).reshape(focal_elements_comb.shape[0], (2**d), num_outputs)
                
                # Calculate min and max for each sublist in all_output
                min_values = np.min(all_output, axis=1) 
                max_values = np.max(all_output, axis=1) 

                lower_bound = np.zeros((num_outputs,len(min_values)))
                upper_bound = np.zeros((num_outputs,len(max_values)))
  
                bounds = np.empty((num_outputs, 2, lower_bound.shape[1])) 

                for i in range(num_outputs):
                    min_values[:,i] = np.sort(min_values[:,i])
                    max_values[:,i]  = np.sort(max_values[:,i])
                    lower_bound[i,:] = min_values[:, i]  # Extract each column
                    upper_bound[i,:] = max_values[:, i]  
                    
                    bounds[i, 0, :] = lower_bound[i,:]
                    bounds[i, 1, :] = upper_bound[i,:]

            case "extremepoints":
                # Determine the positive or negative signs for each input

                res = extremepoints_method(ranges.T, f)
                
                 # Determine the number of outputs from the first evaluation
                try:
                    num_outputs = res.raw_data['sign_x'].shape[0]
                except TypeError:
                    num_outputs = 1  # If f returns a single value

                inpsList = np.zeros((0, d))
                evalsList = np.zeros((0, num_outputs))
                all_output_list = []             

                for out in range(num_outputs):
                    Xsings = np.empty((focal_elements_comb.shape[0], 2, d))
                    current_index = 0
                    for i, slice in enumerate(focal_elements_comb):
                        Xsings[i, :, :] = extreme_pointX(slice, res.raw_data['sign_x'][out])
                        current_index += 1

                        for k in range(Xsings.shape[1]):
                            c = Xsings[i, k, :]
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
                all_output =  np.array(all_output_list).reshape(focal_elements_comb.shape[0], 2, num_outputs)
                
                # Calculate min and max for each sublist in all_output
                min_values = np.min(all_output, axis=1) 
                max_values = np.max(all_output, axis=1) 

                lower_bound = np.zeros((num_outputs,len(min_values)))
                upper_bound = np.zeros((num_outputs,len(max_values)))
  
                bounds = np.empty((num_outputs, 2, lower_bound.shape[1])) 

                for i in range(num_outputs):
                    min_values[:,i] = np.sort(min_values[:,i])
                    max_values[:,i]  = np.sort(max_values[:,i])
                    lower_bound[i,:] = min_values[:, i]  # Extract each column
                    upper_bound[i,:] = max_values[:, i]  
                    
                    bounds[i, 0, :] = lower_bound[i,:]
                    bounds[i, 1, :] = upper_bound[i,:]
                
            case _:
                raise ValueError("Invalid UP method! endpoints_cauchy are under development.")

        results.raw_data['bounds']  = bounds
        results.raw_data['min'] = np.array([{ 'f': lower_bound}])  # Initialize as a NumPy array
        results.raw_data['max'] = np.array([{ 'f': upper_bound}])  # Initialize as a NumPy array

        if save_raw_data == 'yes':
            results.add_raw_data(f= all_output, x= x_combinations)

    elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
        results.add_raw_data(f= None, x= x_combinations)
    
    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")
       
    return results 
