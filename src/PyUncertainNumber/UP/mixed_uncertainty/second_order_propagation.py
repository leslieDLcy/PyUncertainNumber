import numpy as np
from typing import Callable
import tqdm
from PyUncertainNumber.UP.mixed_uncertainty.cartesian_product import cartesian

def second_order_propagation_method(x: list, f:Callable = None, results:dict = None,
                             method = 'endpoints',
                             lim_Q: np.array = np.array([0.01, 0.09]), n_disc:int = 5, 
                             save_raw_data= 'no'):
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
    
    bounds_x = []  
    u_list = []  # List to store 'u' for each uncertain number
    
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

    for i, un in enumerate(x):
        print(f"Processing variable {i + 1} with essence: {un.essence}")

        if un.essence == "distribution":
            # perform outward directed discretisation 
            nd = n_disc + 1  
            distribution_family = un.distribution_parameters[0]

            if distribution_family == 'triang':
                u = np.linspace(0, 1, nd)
            else:
                u = np.linspace(lim_Q[0], lim_Q[1], nd)

        elif un.essence == "pbox":
            distribution_family = un.distribution_parameters[0]
            if distribution_family == 'triang': 
                u = np.linspace(0, 1, n_disc)
            else:
                u = np.linspace(lim_Q[0], lim_Q[1], n_disc)
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

            case "interval":
                temp_xl = np.array([un.bounds[0]])  # Repeat lower bound for all quantiles
                temp_xr = np.array([un.bounds[1]])  # Repeat upper bound for all quantiles               
                rang = np.array([temp_xl, temp_xr]).T  # Create a 2D array of bounds
                bounds_x.append(rang)

            case "pbox":
                temp_xl = un.ppf(u_list[i])[0]  
                temp_xr = un.ppf(u_list[i])[1]               
                rang = np.array([temp_xl, temp_xr]).T  # Create a 2D array of bounds
                bounds_x.append(rang)

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
            case _:
                raise ValueError("Invalid UP method! endpoints_cauchy and subinterval reconstitution under development.")

        results['bounds']= bounds
        results['min']['y']=  lower_bound
        results['max']['y']=  upper_bound

        if save_raw_data == 'yes':
            results['raw_data'] = {'f': all_output, 'x': x_combinations}

    elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
        results['raw_data'] = {'f': None, 'x': x_combinations}
    
    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")

       
    return results 

