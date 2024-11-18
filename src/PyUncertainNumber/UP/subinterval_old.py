import numpy as np
import tqdm
from typing import Callable

def cartesian(*arrays):
    """ Computes the Cartesian product of multiple input arrays

    args:
       - *arrays: Variable number of np.arrays representing the sets of values for each dimension.


    signature:
       - cartesian(*x:np.array)  -> np.ndarray

    note:
       - The data type of the output array is determined based on the input arrays to ensure compatibility.

    return:
        - darray: A NumPy array where each row represents one combination from the Cartesian product. 
                  The number of columns equals the number of input arrays.

    example:
        x = np.array([1, 2], [3, 4], [5, 6])
        y = cartesian(*x)
        # Output: 
        # array([[1, 3, 5],
        #        [1, 3, 6],
        #        [1, 4, 5],
        #        [1, 4, 6],
        #        [2, 3, 5],
        #        [2, 3, 6],
        #        [2, 4, 5],
        #        [2, 4, 6]])

"""
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def subinterval_method(x:np.ndarray, f:Callable, n:np.array, save_raw_data = 'no'):

    """ Performs uncertainty propagation using the Subinterval Reconstitution Method.

    args:
        - x: A 2D NumPy array where each row represents an input variable and the two columns
           define its lower and upper bounds (interval).
        - f: A callable function that takes a 1D NumPy array of input values and returns the
           corresponding output(s).
        - n: A scalar (integer) or a 1D NumPy array specifying the number of subintervals for 
           each input variable. 
            - If a scalar, all input variables are divided into the same number of subintervals.
            - If an array, each element specifies the number of subintervals for the 
              corresponding input variable.
        - save_raw_data: Controls the amount of data returned:
            - 'no': Returns only the minimum and maximum output values along with the 
                   corresponding input values.
            - 'yes': Returns the above, plus the full arrays of unique input combinations 
                     (`all_input`) and their corresponding output values (`all_output`).


    signature:
        subinterval_method(x:np.ndarray, f:Callable, n:np.array, save_raw_data = 'no') -> np.ndarray

    note:
        - The function assumes that the intervals in `x` represent uncertainties with some 
          form of distribution (not necessarily uniform) and aims to provide conservative 
          bounds on the output uncertainty.
        - The computational cost increases exponentially with the number of input variables 
          and the number of subintervals per variable.
        - If the `f` function returns multiple outputs, the `all_output` array will be 2-dimensional.

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
            - INPUT: 2D NumPy array containing all unique combinations of input subinterval 
                     endpoints.
            - OUTPUT: 1D or 2D NumPy array containing the corresponding output values 
                      for each input combination.


    example:
        x = np.array([1, 2], [3, 4], [5, 6])
        fun = lambda x: x[0] + x[1] + x[2]
        n = 2
        miny, maxy, x_miny, x_maxy  = subinterval_method(x, fun, n, save_raw_data ='no')
        # OR
        #miny, maxy, x_miny, x_maxy, all_input, all_output  = subinterval_method(x, fun, n, save_raw_data ='yes')

        miny = array([9.])
        maxy = array([12.])
        x_miny = array([[1., 3., 5.]])
        x_maxy = array([[2., 4., 6.]]))

    # TODO to test the code here,  
    """
    # Create a sequence of values for each interval based on the number of divisions provided 
    # The divisions may be the same for all intervals or they can vary.
    m = x.shape[0]
    if type(n) == int: #All inputs have identical division
        total = (n + 1)**m 
        Xint = np.zeros((0,n + 1), dtype=object )
        for xi in x:
            new_X =  np.linspace(xi[0], xi[1], n+1)
            Xint = np.concatenate((Xint, new_X.reshape(1,n+1)),  axis=0)
    else: #Different divisions per input
        total = 1
        Xint = []   
        for xc, c in zip(x, range(len(n))):
            total *= (n[c]+1)
            new_X =  np.linspace(xc[0], xc[1], n[c]+1)
            
            Xint.append(new_X)
 
        Xint = np.array(Xint, dtype=object) 
    # create an array with the unique combinations of all subintervals 
    X = np.array(cartesian(*Xint), dtype=object) 

    # propagates the epistemic uncertainty through subinterval reconstitution   
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
