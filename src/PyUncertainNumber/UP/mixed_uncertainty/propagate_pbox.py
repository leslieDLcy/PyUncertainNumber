import numpy as np
from typing import Callable
import tqdm
from PyUncertainNumber.UP.mixed_uncertainty.Cartesian import cartesian
from scipy.stats import norm 
from PyUncertainNumber.UC.uncertainNumber import UncertainNumber as UN

import matplotlib.pyplot as plt 

##TODO finish off the method in line with all other methods
##TODO also finish other methods like hte cauchy or the subinterval reconstitution. 
##TODO we can change the d_disc for each distribution. 

def mixed_uncertainty_method(x: list, f:Callable = None, method = 'endpoints',
                             lim_Q: np.array = np.array([0.01, 0.09]), n_disc:int = 5):
    """
    Args:
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
                x_combinations = [] #np.zeros(2**d, d)
                for array in focal_elements_comb:
                    cartesian_product_x = cartesian(*array) 
                    x_combinations.append(cartesian_product_x.tolist()) 
                
                # Initialize all_output as a list to store outputs initially
                all_output_list = []  
                current_index = 0

                for i, array in enumerate(x_combinations):
                    for j, case in enumerate(array):
                        im = np.where((inpsList == case).all(axis=1))[0]
                        if not im.size:
                            output = f(case)
                            all_output_list.append(output)  # Append to the list
                            inpsList = np.vstack([inpsList, case])
                            evalsList.append(output)
                            numRun += 1
                        else:
                            all_output_list.append(evalsList[im[0]])
                        current_index += 1

                # Determine num_outputs AFTER running the function
                try:
                    num_outputs = len(all_output_list[0])
                except TypeError:
                    num_outputs = 1

                # Convert all_output to a 2D NumPy array
                all_output = np.array(all_output_list).reshape((len(x_combinations), -1, num_outputs))
                
                # Calculate min and max for each sublist in all_output
                min_values = np.min(all_output, axis=1) 
                max_values = np.max(all_output, axis=1) 

                for i in range(num_outputs):
                    min_values[:,i] = np.sort(min_values[:,i])
                    max_values[:,i]  = np.sort(max_values[:,i])

                    
                ## Need to check the flatten, i know what it does it is good when you have one column not here. 


    
    return  min_values #Include all_combinations in the return values
    
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

a = mixed_uncertainty_method(x=[L, I, F, E], #['L', 'I', 'F', 'E'], 
          f = cantilever_beam_deflection, 
          n_disc= 3
         )

print(a)
  

    # xl = np.zeros((nd , d))
    # xr = np.zeros((nd, d))
    # ranges = np.zeros((2, d))
    
    # for i, un in enumerate(x):
    #     match un.essence:
            
    #         case "distribution":
    #             # xl[:, i] = un.ppf(u[:-1])  # Calculate ppf for all quantiles at once
    #             # print('xl', xl[:,i])
    #             # xr[:, i] = un.ppf(u[1:])   # Calculate ppf for all quantiles at once
    #             # ranges[:, i] = [xl[0, i], xr[-1, i]]
    #             pass
    #         case 'interval':
    #             # Assign the lower bound to the first row of xl for this UN
    #             xl[0, i] = un.bounds[0]  
    #             # Assign the upper bound to the last row of xr for this UN
    #             xr[-1, i] = un.bounds[1]  
    #             ranges[:, i] = un.bounds
    #         case "pbox":
    #             print("pbox_ppf", un.ppf(u))
    #             xl[:, i] = un.ppf(u)[0]  # Calculate ppf for all quantiles at once
    #             xr[:, i] = un.ppf(u)[1]   # Calculate ppf for all quantiles at once
    #             ranges[:, i] = [xl[0, i], xr[-1, i]]
    #         case _:
    #             raise ValueError(f"Unsupported uncertainty type: {un.essence}")

    # return xl, xr, ranges
    
# A = UN(
#     essence="pbox",
#     distribution_parameters=["gaussian", [[1, 2], [1,2]]]
# )

# B = UN(
#     essence="distribution",
#     distribution_parameters=["gaussian", [1, 1]]
# )

# C = UN(
#     essence="interval",
#     bounds= [3, 4] 
# )

# V = UN(name='testing', 
#         symbol='V', 
#         essence='distribution', 
#         distribution_parameters=["triang", [0.5, 10,20]])

# C.display
# plt.show
# print(A)
# print(B)
# print(C)
#xl, xr, range = mixed_uncertainty(x =[A,B,C], f= None,   n_disc=3)


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


