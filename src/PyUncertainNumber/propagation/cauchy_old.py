from scipy.optimize import brentq
import numpy as np
import tqdm
from typing import Callable

def cauchydeviate_method(x: np.ndarray, f: Callable, n: int, save_raw_data = 'no'):
    """
    Performs uncertainty propagation using the Cauchy Deviate Method. 

    Args:
        x: A 2D NumPy array representing the intervals for each input variable.
            Each inner list or row should contain two elements:
            the lower and upper bounds of the interval.
        fun: A callable function that takes a 1D NumPy array of input values
             and returns a single output value or an array of output values.
        n: The number of samples (Cauchy deviates) to generate for each input variable.

    Returns:
        A 2D NumPy array: the estimated lower and upper bounds of the output
        interval(s), representing the range of possible output values considering
        the input uncertainties.
    """
    if (save_raw_data != 'no'):
        print("Raw data in the Cauchy deviates methods are not tractable")
        
    x = np.array(x).reshape(-1, 2)  # Ensure x is 2D
    lo, hi = x.T  # Unpack lower and upper bounds directly

    xtilde = (lo + hi) / 2
    Delta = (hi - lo) / 2
    results = {}
    
    ytilde = f(xtilde)

    if isinstance(ytilde, (float, np.floating)):
       deltaF = np.zeros(n);
       
       for k in tqdm.tqdm(range(1,n)):
           r = np.random.rand(x.shape[0]);
           c = np.tan(np.pi * (r - 0.5));
           K = np.max(c);
           delta = Delta * c / K;
           x = xtilde - delta;
           deltaF[k] = (K * (ytilde - f(x)));
           
       Z = lambda Del: n/2 - np.sum(1 / (1 + (deltaF / Del)**2));
       zRoot = brentq(Z, 0.0001, max(deltaF)/2)
       min_candidate = ytilde - zRoot
       max_candidate  = ytilde + zRoot 
       
       #return min_candidate, max_candidate
    
    else:
       
        len_y = len(ytilde)
        deltaF = np.zeros((n, len_y));
        min_candidate = np.empty((len_y))
        max_candidate = np.empty((len_y))
        new_f = np.empty((n,len_y))
       
        for k in tqdm.tqdm(range(n)):
            r = np.random.rand(x.shape[0]);
            c = np.tan(np.pi * (r - 0.5));
            K = np.max(c);
            delta = Delta * c / K;
            x = xtilde - delta;
            result =  f(x)
            new_f[k] = np.array(result)
            
            for i in range(len_y):
                deltaF[k,i] = K * (ytilde[i] - new_f[k,i]) 
        
        for i in range(len_y): 
            mask = np.isnan(deltaF[:,i])
            filtered_deltaF_i = deltaF[:, i][~mask]
            
            Z = lambda Del: n/2 - np.sum(1 / (1 + (filtered_deltaF_i / Del)**2));
            zRoot = brentq(Z, 0.0001, max(filtered_deltaF_i)/2)
            min_candidate[i] = ytilde[i] - zRoot
            max_candidate[i] = ytilde[i] + zRoot 
        
    results = {
            'min': {
                'f': min_candidate,
                'x': None
                },
            'max': {
                'f': max_candidate,
                'x': None
               },
            'raw_data': {
                'f': None,
                'x': None
                }
           }
    return results
