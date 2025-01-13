import numpy as np
from typing import Callable, Union
import tqdm
from PyUncertainNumber.propagation.epistemic_uncertainty.cartesian_product import cartesian
from PyUncertainNumber.propagation.epistemic_uncertainty.extreme_point_func import extreme_pointX
from PyUncertainNumber.propagation.epistemic_uncertainty.extremepoints import extremepoints_method
from PyUncertainNumber.propagation.utils import propagation_results, condense_bounds

#TODO add tail concentrating algorithms.
#TODO add x valus for min and max
def second_order_propagation_method(x: list, f:Callable = None,  
                                    results: propagation_results = None, 
                                    method = 'endpoints',
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

        if un.essence != "interval":
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
                temp_xl = un.ppf(u_list[i][:-1])[0]  
                temp_xr = un.ppf(u_list[i][1:])[1]           
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
            case "endpoints" | "second_order_endpoints":
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

                for case in tqdm.tqdm(x_combinations, desc="Evaluating focal points"):  # Wrap the loop with tqdmx_combinations
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

            case "extremepoints"| "second_order_extremepoints":
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

                # Preallocate all_output_list with explicit loops
                all_output_list = []
                for _ in range(num_outputs):
                    output_for_current_output = []
                    for _ in range(len(focal_elements_comb)):  # Changed to focal_elements_comb
                        output_for_current_output.append([None, None])
                    all_output_list.append(output_for_current_output)
                
                for i, slice in tqdm.tqdm(enumerate(focal_elements_comb), desc="Evaluating focal points", total=len(focal_elements_comb)):  # Iterate over focal_elements_comb
                    Xsings = np.empty((2, d))  # Changed to 2 for the two extreme points
                    Xsings[:, :] = extreme_pointX(slice, res.raw_data['sign_x'])  # Use the entire sign_x array

                    for k in range(Xsings.shape[0]):  # 
                        c = Xsings[k, :]
                        im = np.where((inpsList == c).all(axis=1))[0]
                        if not im.size:
                            output = f(c)

                            # Store each output in a separate sublist
                            for out in range(num_outputs):
                                all_output_list[out][i][k] = output[out]  # Changed indexing

                            inpsList = np.vstack([inpsList, c])

                            # Ensure output is always a NumPy array
                            if not isinstance(output, np.ndarray):
                                output = np.array(output)

                            evalsList = np.vstack([evalsList, output])
                        else:
                            for out in range(num_outputs):
                                all_output_list[out][i][k] = evalsList[im[0]][out]  # Changed indexing
 
                # Reshape all_output based on the actual number of elements per output
                all_output = np.array(all_output_list, dtype=object)
                all_output = np.reshape(all_output, (num_outputs, -1, 2))  # Reshape to 3D
                
                # Calculate min and max for each sublist in all_output
                min_values = np.min(all_output, axis=2) 
                max_values = np.max(all_output, axis=2) 

                lower_bound = np.zeros((num_outputs,min_values.shape[1]))
                upper_bound = np.zeros((num_outputs,max_values.shape[1]))
  
                bounds = np.empty((num_outputs, 2, lower_bound.shape[1])) 

                for i in range(num_outputs):
                    lower_bound[i,:] = np.sort(min_values[i,:])  # Extract each column
                    upper_bound[i,:] = np.sort(max_values[i,:])
                    
                    bounds[i, 0, :] = lower_bound[i,:]
                    bounds[i, 1, :] = upper_bound[i,:]
                
            case _:
                raise ValueError("Invalid UP method! endpoints_cauchy are under development.")

        if condensation is not None:
            bounds = condense_bounds(bounds, condensation) 

        results.raw_data['bounds']  = bounds
        results.raw_data['min'] = np.array([{'f': lower_bound[i, :]} for i in range(num_outputs)])  # Initialize as a NumPy array
        results.raw_data['max'] = np.array([{'f': upper_bound[i, :]} for i in range(num_outputs)])  # Initialize as a NumPy array

        if save_raw_data == 'yes':
            print('No raw data provided for this method!')
            #results.add_raw_data(f= all_output, x= x_combinations)

    elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
        results.add_raw_data(f= None, x= x_combinations)
    
    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")
       
    return results 

# from PyUncertainNumber import UncertainNumber

# def myFunctionWithTwoOutputs(x):
#     """
#     Example function with two outputs.
#     Replace this with your actual function logic.
#     """
#     input1= x[0]
#     input2=x[1]
#     input3=x[2]
#     input4=x[3]
#     input5=x[4]
#     output1 = input1 + input2 + input3 + input4 + input5
#     output2 = input1 * input2 * input3 * input4 * input5
#     return np.array([output1])#np.array([output1, output2])

# means = np.array([1, 2, 3, 4, 5])
# stds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

# n_disc = 10  # Number of discretizations

# x = [
#         UncertainNumber(essence = 'distribution', distribution_parameters= ["gaussian",[means[0], stds[0]]]),

#         UncertainNumber(essence = 'interval', bounds= [means[1]-2* stds[1], means[1]+2* stds[1]]),
#         UncertainNumber(essence = 'interval', bounds= [means[2]-2* stds[2], means[2]+2* stds[2]]),
#         UncertainNumber(essence = 'interval', bounds= [means[3]-2* stds[3], means[3]+2* stds[3]]),
#         UncertainNumber(essence = 'interval', bounds= [means[4]-2* stds[4], means[4]+2* stds[4]])
#         #UncertainNumber(essence = 'distribution', distribution_parameters= ["gaussian",[means[1], stds[1]]]),
#         # UncertainNumber(essence = 'distribution', distribution_parameters= ["gaussian",[means[2], stds[2]]]),
#         # UncertainNumber(essence = 'distribution', distribution_parameters= ["gaussian",[means[3], stds[3]]]),
#         # UncertainNumber(essence = 'distribution', distribution_parameters= ["gaussian",[means[4], stds[4]]]),
#     ]
        
        
       

# results = second_order_propagation_method(x=x, f=myFunctionWithTwoOutputs, method = 'endpoints', n_disc= 100)
    
# import matplotlib.pyplot as plt

# def plotPbox(xL, xR, p=None):
#     """
#     Plots a p-box (probability box) using matplotlib.

#     Args:
#         xL (np.ndarray): A 1D NumPy array of lower bounds.
#         xR (np.ndarray): A 1D NumPy array of upper bounds.
#         p (np.ndarray, optional): A 1D NumPy array of probabilities corresponding to the intervals.
#                                    Defaults to None, which generates equally spaced probabilities.
#         color (str, optional): The color of the plot. Defaults to 'k' (black).
#     """
#     xL = np.squeeze(xL)  # Ensure xL is a 1D array
#     xR = np.squeeze(xR)  # Ensure xR is a 1D array

#     if p is None:
#         p = np.linspace(0, 1, len(xL))  # p should have one more element than xL/xR

#     if p.min() > 0:
#         p = np.concatenate(([0], p))
#         xL = np.concatenate(([xL[0]], xL))
#         xR = np.concatenate(([xR[0]], xR))

#     if p.max() < 1:
#         p = np.concatenate((p, [1]))
#         xR = np.concatenate((xR, [xR[-1]]))
#         xL = np.concatenate((xL, [xL[-1]]))
    
#     colors = 'black'
#     # Highlight the points (xL, p)
#     plt.scatter(xL, p, color=colors, marker='o', edgecolors='black', zorder=3)

#     # Highlight the points (xR, p)
#     plt.scatter(xR, p, color=colors, marker='o', edgecolors='black', zorder=3)


#     plt.fill_betweenx(p, xL, xR, color=colors, alpha=0.5)
#     plt.plot( [xL[0], xR[0]], [0, 0],color=colors, linewidth=3)
#     plt.plot([xL[-1], xR[-1]],[1, 1],  color=colors, linewidth=3)
#     plt.show()

# results.print()


# plotPbox(results.raw_data['min'][0]['f'], results.raw_data['max'][0]['f'], p=None)
# plt.show()

# # plotPbox(results.raw_data['min'][1]['f'], results.raw_data['max'][1]['f'], p=None)
# # plt.show()

