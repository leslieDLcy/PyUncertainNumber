from scipy.stats import qmc
import numpy as np
import tqdm
from typing import Callable

def sampling_method(x:np.ndarray, f:Callable, n:int, method ='montecarlo', endpoints=False, save_raw_data = 'no'):
    """ Performs uncertainty propagation through sampling-based methods.

    args:
        - x: A 2D NumPy array where each row represents an input variable and the two columns 
           define its lower and upper bounds (interval).
        - fun: A callable function that takes a 1D NumPy array of input values and returns the 
           corresponding output(s).
        - n: The number of samples to generate for the chosen sampling method.
        - method: The sampling method to use. Choose from:
            - 'montecarlo': Monte Carlo sampling (random sampling from uniform distributions)
            - 'lhs': Latin Hypercube sampling (stratified sampling for better space coverage)
        - endpoints: If True, include the interval endpoints in the sampling. Defaults to False.
        - method: 'montecarlo' or 'lhs'
        - save_raw_data: Controls the amount of data returned:
            - 'no': Returns only the minimum and maximum output values along with the 
                   corresponding input values.
            - 'yes': Returns the above, plus the full arrays of input samples (`all_input`) and 
                     output values (`all_output`).


    signature:
        sampling_method(x:np.ndarray, f:Callable, n:int, method ='montecarlo', endpoints=False, save_raw_data = 'no') -> np.ndarray

    note:
        - The function assumes that the intervals in `x` represent uniform distributions.
        - The implementation for including endpoints (`endpoints=True`) is not provided 
          and needs to be added based on specific requirements.
        - If the `f` function returns multiple outputs, the `all_output` array will be 2-dimensional y and x for all x samples.

    returns:
        - If `save_raw_data == 'no'`: 
            - min_candidate: 1D NumPy array containing the minimum output value(s).
            - max_candidate: 1D NumPy array containing the maximum output value(s).
            - x_miny: 2D NumPy array where each row represents the input values that produced 
                      the corresponding minimum output value.
            - x_maxy: 2D NumPy array wher e each row represents the input values that produced 
                      the corresponding maximum output value.
        - If `save_raw_data == 'yes'`: 
            - The above four arrays, plus:
            - all_input: 2D NumPy array containing all generated input samples.
            - all_output: 1D or 2D NumPy array containing the corresponding output values 
                      for each input sample.


    example:
        x = np.array([1, 2], [3, 4], [5, 6])
        fun = lambda x: x[0] + x[1] + x[2]
        n = 5
        miny, maxy, x_miny, x_maxy  = sampling_method(x fun, n, method ='montecarlo', endpoints=False, save_raw_data = 'no')
        miny, maxy, x_miny, x_maxy, all_input, all_output  = sampling_method(x fun, n, method ='montecarlo', endpoints=False, save_raw_data = 'yes')

    #TODO test it in this platform and see what to be done for the endpoints= True, 
    # it is valid only for monotonic functions, if there are more than one input with the same min or max it cannot cope, i could highlight that in the postprocessing stage.
    """    
    # x = np.array(x).reshape(-1,2) #Just in case
    m = x.shape[0]
    total = n
    
    lo = x.T[0]
    hi = x.T[1]

    if method == 'montecarlo':
        X = np.random.rand(n, m)
    elif method == 'lhs':
        sampler = qmc.LatinHypercube(m)
        X = sampler.random(n)
        
    if endpoints:
        pass
        
    X = lo + (hi-lo) * X
    
    # propagates the epistemic uncertainty through sampling methods   
    if (save_raw_data == 'no'):
        j = 0
        new_f = f(X[j])
        
        if isinstance(new_f, float) or isinstance(new_f, np.float64):
            new_f = np.full((1,), new_f)
            len_y = 1
        else:
            len_y = len(new_f)
  
        len_y = len(new_f)

        max_candidate = np.full((len_y), -np.inf) 
        min_candidate = np.full((len_y),  np.inf)
        x_maxy = np.zeros((len_y,m))
        x_miny = np.zeros((len_y,m))
    
        for y_i in range(len_y):
            if new_f[y_i] > max_candidate[y_i]:
                max_candidate[y_i] = new_f[y_i]
                x_maxy[y_i] = X[j]
            if new_f[y_i] < min_candidate[y_i]:
                min_candidate[y_i] =  new_f[y_i]
                x_miny[y_i] = X[j]
            
        for j in tqdm.tqdm(range(1,total)):
            new_f = f(X[j])
         
            if isinstance(new_f, float) or isinstance(new_f, np.float64):
                new_f = np.full((1,), new_f)
                len_y = 1
            else:
                len_y = len(new_f)

            for y_i in range(len_y):
                if new_f[y_i] > max_candidate[y_i]:
                    max_candidate[y_i] = new_f[y_i]
                    x_maxy[y_i] = X[j]
                if new_f[y_i] < min_candidate[y_i]:
                    min_candidate[y_i] =  new_f[y_i]
                    x_miny[y_i] = X[j]

        return min_candidate, max_candidate, x_miny, x_maxy
    else:
        j = 0
        new_f = f(X[j])
        
        if isinstance(new_f, float) or isinstance(new_f, np.float64):
            new_f = np.full((1,), new_f)
            len_y = 1
        else:
            len_y = len(new_f)
  
        if isinstance(new_f, np.ndarray) == False:
            new_f = np.array(new_f, dtype = object)
        
        len_y = len(new_f)
        
        all_output = new_f.reshape(1,len_y)
        all_input = X

        max_candidate = np.full((len_y), -np.inf) 
        min_candidate = np.full((len_y),  np.inf)
        x_maxy = np.zeros((len_y,m))
        x_miny = np.zeros((len_y,m))
    
        for y_i in range(len_y):
            if new_f[y_i] > max_candidate[y_i]:
                max_candidate[y_i] = new_f[y_i]
                x_maxy[y_i] = X[j]
            if new_f[y_i] < min_candidate[y_i]:
                min_candidate[y_i] =  new_f[y_i]
                x_miny[y_i] = X[j]
            
        for j in tqdm.tqdm(range(1,total)):
            new_f = f(X[j])
         
            if isinstance(new_f, float) or isinstance(new_f, np.float64):
                new_f = np.full((1,), new_f)
                len_y = len(new_f)
            else:
                len_y = len(new_f)
            
            if isinstance(new_f, np.ndarray) == False:
                new_f = np.array(new_f, dtype = object)
   
            all_output = np.concatenate((all_output, new_f.reshape(1,len_y)), axis=0)

            for y_i in range(len_y):
                if new_f[y_i] > max_candidate[y_i]:
                    max_candidate[y_i] = new_f[y_i]
                    x_maxy[y_i] = X[j]
                if new_f[y_i] < min_candidate[y_i]:
                    min_candidate[y_i] =  new_f[y_i]
                    x_miny[y_i] = X[j]
        
        return min_candidate, max_candidate, x_miny, x_maxy, all_input, all_output   
