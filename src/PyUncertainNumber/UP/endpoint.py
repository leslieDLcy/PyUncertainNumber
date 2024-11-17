# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:51:46 2024

I have updated Marco's endpoint method to be used for funcitosn which yield more than one ys and asks the user how much data they need to see. 
Works mostly with numpy.

@author: ioanna, Leslie
"""
import numpy as np
import tqdm
from typing import Callable

def index_to_bool_(index:np.ndarray,dim=2):

    """ Converts a vector of indices to an array of boolean pairs for masking.

  Args:
    index: A NumPy array of integer indices representing selected elements or categories. 
           The values in `index` should be in the range [0, dim-1].
    dim: The number of categories or dimensions in the output boolean array. Defaults to 2  

  signature:
    index_to_bool_(index:np.ndarray,dim=2) -> tuple

   note:
        - the augument `index` is an np.ndaray of the index of intervals.
        - the argument `dim` will specify the function mapping of variables to be propagated. 
    
    #If dim > 2,  e.g. (2,0,1,0) the array of booleans is [(0,0,1),(1,0,0),(0,1,0),(1,0,0)].

  Returns:
    - A NumPy array of boolean pairs representing the mask.
  """
    
    index = np.asarray(index,dtype=int)
    return np.asarray([index==j for j in range(dim)],dtype=bool)


def endpoints_method(x:np.ndarray,f:Callable, save_raw_data = 'no'):
    """Performs uncertainty propagation using the Vertex/Endpoints Method.

    This function estimates the range of possible output values for a given function `f` 
    when its input variables have uncertainties represented as intervals in a NumPy array `x`. 
    It systematically evaluates the function at all combinations of interval endpoints (vertices) 
    to determine the minimum and maximum output values.

    args:
        - x: A 2D NumPy array where each row represents an input variable and the two columns
           define its lower and upper bounds (interval).
        - f: A callable function that takes a 1D NumPy array of input values and returns the
           corresponding output(s).
        - save_raw_data: Controls the amount of data returned:
            - 'no': Returns only the minimum and maximum output values along with the 
                   corresponding input values that produced them.
            - 'yes': Returns the above, plus the full arrays of all endpoint combinations 
                     (`all_input`) and their corresponding output values (`all_output`).


    signature:
        Endpoints_Method(x:np.ndarray,f:Callable, save_raw_data = 'yes') -> np.ndarray

    note:
        - This method assumes that the function `f` is monotonic within the given intervals. 
          If this assumption is not met, the results may not accurately represent the true
          range of possible output values. 
        - The computational cost of this method increases exponentially with the number of 
          input variables (`2**m` evaluations, where `m` is the number of variables).
        - If the `f` function returns multiple outputs, the `OUTPUT` array will be 2-dimensional.ombinations (x,y) for 'yes'

    return:
        - If `save_raw_data == 'no'`: 
            - min_candidate: 1D NumPy array containing the minimum output value(s).
            - max_candidate: 1D NumPy array containing the maximum output value(s).
            - x_miny: 2D NumPy array where each row represents the input values that produced 
                      the corresponding minimum output value.
            - x_maxy: 2D NumPy array where each row represents the input values that produced 
                      the corresponding maximum output value.
        - If `save_raw_data == 'yes'`: 
            - The above four arrays, plus:
            - all_input: 2D NumPy array containing all combinations of input interval endpoints.
            - all_output: 1D or 2D NumPy array containing the corresponding output values 
                      for each endpoint combination.


    example:
        x = np.array([1, 2], [3, 4], [5, 6])
        fun = lambda x: x[0] + x[1] + x[2]
        n = 2
        miny, maxy, x_miny, x_maxy  = Endpoints_Method(x, fun, save_raw_data ='no')
        miny, maxy, x_miny, x_maxy, all_input, all_output  = Endpoints_Method(x, fun, save_raw_data ='yes')

    #TODO test it in this platform 
    """

    # Computes the min and max of a monotonic function with endpoints propagation
    # "x has shape (n,2)."
    m=x.shape[0]

    Total = 2**m
    
    if (save_raw_data == 'no'):
        j = 0
        index = tuple([j//2**h-(j//2**(h+1))*2 for h in range(m)]) # tuple of 0s and 1s
        itb = index_to_bool_(index).T
        new_f = f(x[itb])
        print(type(new_f))

        if isinstance(new_f, float) or isinstance(new_f, np.float64):
            new_f = np.full((1,), new_f)
            len_y = 1
        else:
            len_y = len(new_f)

        max_candidate = np.full(( len_y ), -np.inf) 
        min_candidate = np.full(( len_y ),  np.inf)
        x_maxy = np.zeros(( len_y ,m))
        x_miny = np.zeros(( len_y ,m))
    
        for y_i in range( len_y ):
            if new_f[y_i] > max_candidate[y_i]:
                max_candidate[y_i] = new_f[y_i]
                x_maxy[y_i] = x[itb]
            if new_f[y_i] < min_candidate[y_i]:
                min_candidate[y_i] =  new_f[y_i]
                x_miny[y_i] = x[itb]
            
        for j in tqdm.tqdm(range(1,Total)):
            index = tuple([j//2**h-(j//2**(h+1))*2 for h in range(m)]) # tuple of 0s and 1s
            itb = index_to_bool_(index).T
            
            new_f = f(x[itb])
            if isinstance(new_f, float) or isinstance(new_f, np.float64):
                new_f = np.full((1,), new_f)
                len_y = 1
            else:
                len_y = len(new_f)
                
            for y_i in range( len_y ):
                if new_f[y_i] > max_candidate[y_i]:
                    max_candidate[y_i] = new_f[y_i]
                    x_maxy[y_i] = x[itb]
                if new_f[y_i] < min_candidate[y_i]:
                    min_candidate[y_i] =  new_f[y_i]
                    x_miny[y_i] = x[itb]

        return min_candidate, max_candidate, x_miny, x_maxy
    else:
       j = 0
       index = tuple([j//2**h-(j//2**(h+1))*2 for h in range(m)]) # tuple of 0s and 1s
       itb = index_to_bool_(index).T
       new_f = f(x[itb])
       
       if isinstance(new_f, float) or isinstance(new_f, np.float64):
           new_f = np.full((1,), new_f)
           len_y = 1
       else:
           len_y = len(new_f)
           
       if isinstance(new_f, np.ndarray) == False:
           new_f = np.array(new_f, dtype = object)
       
       len_y = len(new_f)
       
       all_output = new_f.reshape(1,len_y)
       all_input = x[itb].reshape(1,m)

       max_candidate = np.full((len_y), -np.inf) 
       min_candidate = np.full((len_y),  np.inf)
       x_maxy = np.zeros((len_y,m)) 
       x_miny = np.zeros((len_y,m)) 

       for y_i in range(len_y):
           if new_f[y_i] > max_candidate[y_i]:
               max_candidate[y_i] = new_f[y_i]
               x_maxy[y_i] = x[itb]
           if new_f[y_i] < min_candidate[y_i]:
               min_candidate[y_i] =  new_f[y_i]
               x_miny[y_i] = x[itb]
           
       for j in tqdm.tqdm(range(1,Total)):
           index = tuple([j//2**h-(j//2**(h+1))*2 for h in range(m)]) # tuple of 0s and 1s
           itb = index_to_bool_(index).T
           new_f = f(x[itb])
           if isinstance(new_f, float) or isinstance(new_f, np.float64):
               new_f = np.full((1,), new_f)
               len_y = 1
           else:
               len_y = len(new_f)
               
           if isinstance(new_f, np.ndarray) == False:
               new_f = np.array(new_f, dtype = object)
  
           all_output = np.concatenate((all_output, new_f.reshape(1,len_y)), axis=0)
           all_input  = np.concatenate((all_input, x[itb].reshape(1,m)), axis=0)

           for y_i in range(len_y):
               if new_f[y_i] > max_candidate[y_i]:
                   max_candidate[y_i] = new_f[y_i]
                   x_maxy[y_i] = x[itb]
                   
               if new_f[y_i] < min_candidate[y_i]:
                   min_candidate[y_i] =  new_f[y_i]
                   x_miny[y_i] = x[itb]     

       return min_candidate, max_candidate, x_miny, x_maxy, all_input, all_output 
