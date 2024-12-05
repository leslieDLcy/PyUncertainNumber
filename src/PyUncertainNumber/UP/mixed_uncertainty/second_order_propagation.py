import numpy as np
from typing import Callable
import tqdm
from PyUncertainNumber.UP.mixed_uncertainty.cartesian_product import cartesian
from scipy.stats import norm 
from PyUncertainNumber.UC.uncertainNumber import UncertainNumber as UN

import matplotlib.pyplot as plt 

##TODO also finish other methods like hte cauchy or the subinterval reconstitution, endpoint_extrempoint. 
##TODO we can change the d_disc for each distribution. 

def second_order_propagation_method(x: list, f:Callable = None, 
                             method = 'endpoints',
                             lim_Q: np.array = np.array([0.01, 0.09]), n_disc:int = 5, 
                             save_raw_data= 'no'):
    """
    args:
      x (list): A list of UncertainNumber objects, each representing an input 
                  variable with its associated uncertainty.
      f (Callable): A callable function that takes a 1D NumPy array of input 
                      values and returns the corresponding output(s). Can be None, 
                      in which case only samples are generated.
      - n_disc (int): The number of horizontal slices for a precise distribution or p-box.
              The same is assumed for all input.
      signs: A NumPy array of signs for the extreme point method.
    notes:
     - Propagates mixed uncertainty types through a black-box function using endpoint method.
    Returns:
      dict: A dictionary containing the results:
            - 'p-box': an emprical p-box will be provided for each output of f

    example:
        #TODO to add an example for this method.
    """
    d = len(x) # dimension of uncertain numbers 
    
    bounds_x = []  
    u_list = []  # List to store 'u' for each uncertain number

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

                for case in x_combinations:  # Iterate directly over the rows of x_combinations
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
    
# example
# from PyUncertainNumber import UncertainNumber as UN
def cantilever_beam_deflection(x):
    """Calculates deflection and stress for a cantilever beam.

    Args:

        x (np.array): Array of input parameters:
            x[0]: Length of the beam (m)
            x[1]: Second moment of area (mm^4)
            x[2]: Applied force (N)
            x[3]: Young's modulus (MPa)

    Returns:
        float: deflection (m)
               Returns np.nan if calculation error occurs.
    """

    beam_length = x[0]
    I = x[1]
    F = x[2]
    E = x[3]
    try:  # try is used to account for cases where the input combinations leads to error in fun due to bugs
        stress = F * beam_length**3 / (3 * E * 10**6 * I) *100 # deflection in m
        deflection = F * beam_length**3 / (3 * E * 10**6 * I)  # deflection in m
    except:
        deflection = np.nan
        stress = np.nan

    return np.array([deflection,stress])

#L = UN(name='beam length', symbol='L', units='m', essence='distribution', distribution_parameters=["gaussian", [10.05, 0.033]])
#I = UN(name='moment of inertia', symbol='I', units='m', essence='distribution', distribution_parameters=["gaussian", [0.000454, 4.5061e-5]])
F = UN(name='vertical force', symbol='F', units='kN', essence='distribution', distribution_parameters=["gaussian", [24, 8.67]])
#I = UN(name='moment of inertia', symbol='I', units='m', essence='interval', bounds=["gaussian", [0.000454, 4.5061e-5]])

E = UN(name='elastic modulus', symbol='E', units='GPa', essence='distribution', distribution_parameters=["gaussian", [210, 6.67]])
L = UN(name='beam length', symbol='L', units='m', essence='interval', bounds= [9.95, 10.05])
I = UN(name='moment of inertia', symbol='I', units='m', essence='interval', bounds= [0.0003861591, 0.0005213425])

a = second_order_propagation_method(x=[L, I, F, E], #['L', 'I', 'F', 'E'], 
          f = cantilever_beam_deflection, 
          n_disc= 3
         )

print(a['bounds'])

#print(xr)
#print(range)

    # # Efficiency upgrade: store repeated evaluations
    # inpsList = np.zeros((0, d))
    # evalsList = np.zeros((0, 2))
    # numRun = 0

    # # Generate Cartesian product and propagate
    # clD = np.full((nd - 1, 2, d), np.nan)
    # cmD = np.full((nd - 1, 2, d), np.nan)
    # for i in range(d):  # First-order propagation
    #     X = ranges.copy()
    #     for j in range(nd - 1):
    #         X[:, i] = [xl[j, i], xr[j, i]]

    #         # Extreme point method
    #         XsignCl = extreme_pointX(X, signs[:, 0])
    #         XsignCm = extreme_pointX(X, signs[:, 1])
    #         C = np.vstack([XsignCl, XsignCm])

    #         cl = np.full(C.shape[0], np.nan)
    #         cm = np.full(C.shape[0], np.nan)
    #         for k in range(C.shape[0]):
    #             c = C[k]
    #             print(f'Input: {i + 1}, FE: {j + 1}, run: {k + 1} -->  ', end='')

    #             # Check for existing evaluations
    #             im = np.where((inpsList == c).all(axis=1))[0]
    #             if not im.size:
    #                 # Replace the following line with your actual xfoilrun function call
    #                 cl[k], cm[k] = simulate_xfoilrun(c)  
    #                 inpsList = np.vstack([inpsList, c])
    #                 evalsList = np.vstack([evalsList, [cl[k], cm[k]]])
    #                 numRun += 1
    #                 print(f'number of evaluations: {numRun}')
    #             else:
    #                 cl[k] = evalsList[im[0], 0]
    #                 cm[k] = evalsList[im[0], 1]
    #                 print('evaluation exists - skipping...')

    #         clD[j, :, i] = [np.min(cl), np.max(cl)]
    #         cmD[j, :, i] = [np.min(cm), np.max(cm)]
    #         print()

    #     clD[:, :, i] = np.array(sorted(clD[:, :, i], key=lambda x: x[0]))  # Sort by first column
    #     cmD[:, :, i] = np.array(sorted(cmD[:, :, i], key=lambda x: x[0]))
    #     print('\n\n')

    # clI = np.zeros((nd - 1, 2))
    # cmI = np.zeros((nd - 1, 2))
    # for i in range(nd - 1):
    #     clI[i] = imp(np.transpose(clD[i, :, :], (1, 0)))
    #     cmI[i] = imp(np.transpose(cmD[i, :, :], (1, 0)))

    # print(f'Total number of evaluations: {numRun}')
    # return clI, cmI, clD, cmD

# def imp(X):
#     """Imposition of intervals."""
#     return np.array([np.max(X[0, :]), np.min(X[1, :])])

# # Placeholder for your xfoilrun function
# def simulate_xfoilrun(c):
#     """Simulates the xfoilrun function."""
#     # Replace this with your actual xfoilrun function or a suitable simulation
#     cl = c[0] + 2 * c[1] - 0.5 * c[2]  # Example calculation for cl
#     cm = c[1] * c[2] + c[3] / c[4]  # Example calculation for cm
#     return cl, cm


