import subprocess
import numpy as np
import pandas as pd
import os
from pathlib import Path
import time

from Iterative_Filtering import iterative_filter

def NASA_UQ_Function(input_file_path, nsamples, nconditions):
    # This section needs to be written as a function that executes the function or model being used
    exe_path = os.path.abspath(Path(__file__).parent / "local_model_windows.exe")  # Path to the executable in the current folder
    command = [exe_path, input_file_path]
    print('Simulation executable has been called.')
    result = subprocess.run(command, capture_output=True, text=True)
    output_file_path = 'Y_out.csv'
    df = pd.read_csv(output_file_path, header=None)
    print(f'Simulation output data loaded from {output_file_path}')

    sample_indices = df[6].unique()  # Extract unique sample indices
    num_samples = len(sample_indices)
    df = df.drop(columns=[6])  # Drop the sample index column
    # df.to_csv(f'./{folder}/Y_out_Level{Iteration}_{repetitions}.csv',header=False,index=False) #stores the full output file (optional)
    Y_out = df.to_numpy().reshape(num_samples, 60, 6).transpose(1, 2, 0)

    epsilon = 1e-14 #Small constant to avoid division by zero
    y4y5 = (Y_out[10,3,:]/(Y_out[10,4,:]+epsilon)) #Get the simulation y ratios for all samples.
    y4y6 = (Y_out[10,3,:]/(Y_out[10,5,:]+epsilon))
    y5y6 = (Y_out[10,4,:]/(Y_out[10,5,:]+epsilon))
    
    y4y5 = np.reshape(y4y5,[nsamples,nconditions]) #Reshape each y threshold matrix into one where all conditions are in one line.
    y4y6 = np.reshape(y4y6,[nsamples,nconditions])
    y5y6 = np.reshape(y5y6,[nsamples,nconditions])

    y_out = np.empty([3,nsamples,nconditions])
    y_out[0] = y4y5
    y_out[1] = y4y6
    y_out[2] = y5y6

    return y_out

'''Base and first query example'''

variables = {
    'names': ['a1','a2','e1','e2','e3','c1','c2','c3','s'],
    'number_type': ['float','float','float','float','float','float','float','float','int'],
    'b': [[0,1],[0,1],[0,1],[0,1],[0,1],[0.533,0.533],[0.666,0.666],[0.5,0.5],[0,1000000]],
    's1': [[0,1],[0,1],[0,1],[0,1],[0,1],[0.052632,0.052632],[0.421053,0.421053],[0.631579,0.631579],[0,1000000]]
}

conditions = {
    'names': ['b','s1'], #condition vector names (for saving files and getting how many conditions are used)
    'values': [[2.031260556,1.746109960],  #y4y5 ratio values
                 [2.408021879,3.429627891],  #y4y6 ratio values
                 [1.185481534,1.964153444]]  #y5y6 ratio values
}

#True epistemic values to aim for: 0.335, 0.596, 0.375

folder = './Samples_15_09_b' #Change as needed for new runs
# os.mkdir(folder)

thresholds = [0.5,0.5,0.3,0.2,0.1,0.05] #Arbitrary ratio list

tic = time.perf_counter()
iterative_filter(NASA_UQ_Function,variables,500,thresholds,conditions,folder,index=[2,3,4],controls=[5,6,7])
toc = time.perf_counter()
print(toc-tic)

'''Base and all queries example'''

# conditions = {
#     'names': ['b','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10'], #condition vector names (for saving files and getting how many conditions are used)
#     'c_values':[[0.533,0.052632,0.157895,0.172571,0.374791,0.673469,0.122449,0.428571,0.99,0.324,0.128], #c1 values
#                 [0.666,0.421053,0.368421,0.515178,0.949813,0.836735,0.306122,0.632653,0.01,0.519,0.795], #c2 values
#                 [0.500,0.631579,0.578947,0.691462,0.731530,0.666667,0.666667,1,0.99,0.703,0.666]],       #c3 values

#     'values': [[2.031260556,1.746109960,1.429010509,1.905941033,0.746474687,1.406255580,1.272302519,1.386273554,1.168335506,1.539947612,1.530131060],  #y4y5 ratio values
#                [2.408021879,3.429627891,2.740270433,3.170222507,1.626907525,2.960595734,2.873171858,2.601963877,4.630457007,2.285175253,3.057806562],  #y4y6 ratio values
#                [1.185481534,1.964153444,1.917599917,1.663337140,2.179454379,2.105304168,2.258245830,1.876948363,3.963293919,1.483930515,1.998395198]], #y5y6 ratio values
# }

# variables = {
#     'names': ['a1','a2','e1','e2','e3','c1','c2','c3','s'],
#     'number_type': ['float','float','float','float','float','float','float','float','int'],
#     'b':  [[0,1],[0,1],[0,1],[0,1],[0,1],[0.533,0.533],[0.666,0.666],[0.5,0.5],[0,1000000]],
#     's1': [[0,1],[0,1],[0,1],[0,1],[0,1],[0.052632,0.052632],[0.421053,0.421053],[0.631579,0.631579],[0,1000000]],
#     's2': [[0,1],[0,1],[0,1],[0,1],[0,1],[0.157895,0.157895],[0.368421,0.368421],[0.578947,0.578947],[0,1000000]],
#     's3': [[0,1],[0,1],[0,1],[0,1],[0,1],[0.172571,0.172571],[0.515178,0.515178],[0.691462,0.691462],[0,1000000]],
#     's4': [[0,1],[0,1],[0,1],[0,1],[0,1],[0.374791,0.374791],[0.949813,0.949813],[0.731530,0.731530],[0,1000000]],
#     's5': [[0,1],[0,1],[0,1],[0,1],[0,1],[0.673469,0.673469],[0.836735,0.836735],[0.666667,0.666667],[0,1000000]],
#     's6': [[0,1],[0,1],[0,1],[0,1],[0,1],[0.122449,0.122449],[0.306122,0.306122],[0.666667,0.666667],[0,1000000]],
#     's7': [[0,1],[0,1],[0,1],[0,1],[0,1],[0.428571,0.428571],[0.632653,0.632653],[1, 1],[0,1000000]],
#     's8': [[0,1],[0,1],[0,1],[0,1],[0,1],[0.99,0.99],[0.01,0.01],[0.99,0.99],[0,1000000]],
#     's9': [[0,1],[0,1],[0,1],[0,1],[0,1],[0.324,0.324],[0.519,0.519],[0.703,0.703],[0,1000000]],
#     's10':[[0,1],[0,1],[0,1],[0,1],[0,1],[0.128,0.128],[0.795,0.795],[0.666,0.666],[0,1000000]],
# }

# folder = './Samples_10_07_all_conditions' #Change as needed for new runs
# os.mkdir(folder)

# radii = [0.5,0.5,0.3,0.2,0.1,0.05,0.03,0.02] #Arbitrary ratio list

# tic = time.perf_counter()
# iterative_filter(NASA_UQ_Function,variables,2000,thresholds,conditions,folder,index=[2,3,4],controls=[5,6,7])
# toc = time.perf_counter()
# print(toc-tic)