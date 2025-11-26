"""
Solution to the scenario optimization problem for enclosing sets. 

A scenario optimization inputs (1) a table of observations i.e. an (nxd) array, and (2) a shape. 

In general, a scenario program should output three data structures:

(1) A list of integers pointing to the support vectors in the data set;
(2) A data structure representing the set if parametric, e.g. a box, a circle, an ellipse.
(3) A function or object representing the optimal set. For example, the function/method returns true if evaluated inside the set and false otherwise. 

"""

import numpy as numpy
import numpy.matlib as matlib
import scipy.special as sp

def minimal_enclosing_hyperbox(x):
    """
    Inputs
    x: (nxd) array, where n is the size of the data, and d is the number of dimensions of the box

    Outputs
    active_scenarios: list[int], a list of integers pointing to the corresponding active scenarios in the dataset
    box: 2xd array, an enclosing box of dimension d.
    """
    x = numpy.asarray(x,dtype=float) # coherce any iterable to numpy array
    n,d = x.shape
    index,box = [],[]
    for j in range(d):
        s = numpy.argsort(x[:,j])
        index += [s[0],s[-1]]
        box.append([x[s[0],j], x[s[-1],j]])
    active_scenarios = list(set(index)) # list of indices corresponding to the active scenarios
    return active_scenarios, numpy.asarray(box,dtype=float)


def is_inside_box(x,abox):
    """
    x:    (nxd) array
    abox: (dx2) array, or list[list[2 float]] ex.: [[0,1],[3,8],[1,9]] <- a.k.a. interval iterable
    """
    n,d = x.shape
    box = numpy.asarray(abox, dtype=float)
    box_lo = box[:,0]
    box_hi = box[:,1]
    inside = (matlib.repmat(box_lo,n,1) <= x) & (x <= matlib.repmat(box_hi,n,1))
    return numpy.all(inside,axis=1)


triu,tril,log,exp,repmat = numpy.triu, numpy.tril, numpy.log, numpy.exp, numpy.matlib.repmat

def epsLU(k,N,bet):
    """
    Port of the MATLAB code provided by the Authors.

    %% Reference Article 
    % Title Risk and complexity in scenario optimization 
    % Authors S. Garatti and ·M. C. Campi
    % Journal Mathematical Programming 
    % DOI https://doi.org/10.1007/s10107-019-01446-4

    % This function provide the lower and upper reliability parameter for a
    % convex program as defined in Eq (14) of the refernced paper

    % N= number of samples
    % k = Number of scenarios \delta_i for which \zeta_i \geq 0, i.e.  f(x,\delta_i) \geq 0
    % beta = confidence parameter (e.g. very high confidence beta=10^-8)

    """
    alphaL = sp.betaincinv(k,N-k+1,bet)
    alphaU = 1 - sp.betaincinv(N-k+1,k,bet)

    m1 = numpy.asarray(range(k,N+1),dtype=int) # m1 = [k:1:N]
    # ones = numpy.ones((N-k+1,1))
    aux1 = numpy.sum(triu(log(repmat(m1,N-k+1,1)),k=1),axis=1) # aux1 = sum(triu(log(ones(N-k+1,1)*m1),1),2); 
    aux2 = numpy.sum(triu(log(repmat(m1-k,N-k+1,1)),k=1),axis=1) # aux2 = sum(triu(log(ones(N-k+1,1)*(m1-k)),1),2); 
    coeffs1 = aux2-aux1
    m2 = numpy.asarray(range(N+1,4*N+1),dtype=int) # m2 = [N+1:1:4*N]; 
    aux3 = numpy.sum(tril(log(repmat(m2,3*N,1)),k=0),axis=1) # aux3 = sum(tril(log(ones(3*N,1)*m2)),2);
    aux4 = numpy.sum(tril(log(repmat(m2-k,3*N,1)),k=0),axis=1) # aux4 = sum(tril(log(ones(3*N,1)*(m2-k))),2);
    coeffs2 = aux3-aux4
    t1 = 1-alphaL
    t2 = 1
    poly1 = 1+bet/(2*N)-bet/(2*N)*sum(exp(coeffs1-(N-m1)*log(t1)))-bet/(6*N) * sum(exp(coeffs2+(m2-N)*log(t1)))
    poly2 = 1+bet/(2*N)-bet/(2*N)*sum(exp(coeffs1-(N-m1)*log(t2)))-bet/(6*N) * sum(exp(coeffs2+(m2-N)*log(t2)))

    if (poly1*poly2) > 0:
        epsL = 0
    else:
        while (t2-t1) > 1e-10:
            t = (t1+t2)/2
            polyt = 1+bet/(2*N)-bet/(2*N)*sum(exp(coeffs1-(N-m1)*log(t)))-bet/(6*N)*sum(exp(coeffs2+(m2-N)*log(t))); 
            if polyt > 0:
                t1=t
            else:
                t2=t
        epsL = 1-t2
    
    t1 = 0
    t2 = 1-alphaU
    poly1 = 1+bet/(2*N)-bet/(2*N)*sum(exp(coeffs1 - (N-m1)*log(t1)))-bet/(6*N)*sum(exp(coeffs2 + (m2-N)*log(t1))); 
    poly2 = 1+bet/(2*N)-bet/(2*N)*sum(exp(coeffs1 - (N-m1)*log(t2)))-bet/(6*N)*sum(exp(coeffs2 + (m2-N)*log(t2))); 
    if (poly1*poly2) > 0:
        epsL = 0
    else:
        while (t2-t1) > 1e-10:
            t = (t1+t2)/2
            polyt =1+bet/(2*N)-bet/(2*N)*sum(exp(coeffs1-(N-m1)*log(t)))-bet/(6*N)*sum(exp(coeffs2 + (m2-N)*log(t)))
            if polyt > 0:
                t2=t
            else:
                t1=t
        epsU = 1-t1; 

    return epsL, epsU




if __name__ == '__main__':
    bet=0.1
    k=2
    N=20
    out = epsLU(k,N,bet)
    print(out)

