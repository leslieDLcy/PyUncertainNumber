###
#   Defines copula functions. For plotting and use in convolutions
#
#   To Do:
#
#           -> Once completed, the Sigma, Tau and Rho convolutions may be defined
#
#
#       By: Ander Gray, University of Liverpool, ander.gray@liverpool.ac.uk
###
if __name__ is not None and "." in __name__:
    from .interval import *
else:
    from interval import *
    
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm

__all__ = ['Copula','ClaGen','ClaInv','FGen','FInv','indep','perf','opp','Cla','F','Gau','pi','M','W','Frank','Clayton','Gaussian']
class Copula(object):

    def __init__(self, cdf=None, func=None, param = None):

        self.cdf = cdf          # Could also include density, however expensive to compute
        self.func = func        # If the functional form is known
        self.param = param      # parameter for func

    def __repr__(self):

        statement1 = "Arbitrary"
        statement2 = ""

        if self.func is not None:
            func = self.func
            if func == indep: func = "π"
            if func == perf: func = "M"
            if func == opp: func = "W"
            if func == Gau: func = "Gau"
            if func == Cla: func = "Clayton"
            if func == F: func = "Frank"
            statement1 = f'{func}'

        if self.param is not None:
            func = self.func
            parName = "par"
            if func == Gau: parName = "r"
            if func == F: parName = "s"
            if func == Cla: parName = "t"
            statement2 = f'{parName}={self.param}'

        return f'Copula ~ {statement1}({statement2})'

    def get_cdf(self, x, y):    # x, y are points on the unit square

        if self.func is not None:   # If function is known, return it
            if self.param is not None: return self.func(x, y, self.param)
            return self.func(x,y)

        else:   # Simple inner/outter interpolation. Returns interval for cdf value
            xsize, ysize = self.cdf.shape

            xIndexLower = int(np.floor(x * (xsize-1)))
            yIndexLower = int(np.floor(y * (ysize-1)))

            xIndexUpper = int(np.ceil(x * (xsize-1)))
            yIndexUpper = int(np.ceil(y * (ysize-1)))

            return Interval(self.cdf[xIndexLower, yIndexLower], self.cdf[xIndexUpper, yIndexUpper])

    def get_mass(self, x, y):   # x, y are intervals on the unit square

        C22 = self.get_cdf(x.hi(), y.hi())
        C21 = self.get_cdf(x.hi(), y.lo())
        C12 = self.get_cdf(x.lo(), y.hi())
        C11 = self.get_cdf(x.lo(), y.lo())

        return C22 - C21 - C12 + C11

    def show(self, pn = 50, fontsize = 20, cols = cm.RdGy):
        ##
        #   All the extra stuff is so that no more than 200 elements are plotted
        ##
        A = self.cdf; m = len(A)

        if m < pn: pn = m

        x = y = np.linspace(0, 1, num = pn)
        X, Y = np.meshgrid(x,y)

        nm = round(m/pn)
        Z = A[::nm,::nm]    # Skip over evelemts

        fig = plt.figure("SurfacPlots",figsize=(10,10))
        ax = fig.add_subplot(1,1,1,projection="3d")
        ax.plot_surface(X, Y, Z, rstride=2,edgecolors="k", cstride=2, alpha=0.8, linewidth=0.25, cmap = cols)
        plt.xlabel("X",fontsize = fontsize); plt.ylabel("Y", fontsize = fontsize)

        plt.show()

    def showContour(self, fontsize = 20, cols = cm.coolwarm):
        ##
        #   All the extra stuff is so that no more than 200 elements are plotted
        ##
        pn = 200    # Max plot number
        A = self.cdf; m = len(A)

        if m < pn: pn = m

        x = y = np.linspace(0, 1, num = pn)
        X, Y = np.meshgrid(x,y)

        nm = round(m/pn)
        Z = A[::nm,::nm]    # Skip over evelemts

        fig = plt.figure("SurfacPlots",figsize=(10,10))
        ax = fig.add_subplot(2,1,1,projection="3d")
        ax.plot_surface(X, Y, Z, rstride=2,edgecolors="k", cstride=2, alpha=0.8, linewidth=0.25, cmap = cols)
        plt.xlabel("X",fontsize = fontsize); plt.ylabel("Y", fontsize = fontsize)
        plt.title("Surface Plot", fontsize = fontsize)

        ax1 = fig.add_subplot(2,1,2)
        cp = ax1.contour(X, Y, Z, cmap = cols, levels = 15)
        ax1.clabel(cp, inline=1, fontsize=10)
        plt.xlabel("X", fontsize = fontsize); plt.ylabel("Y", fontsize = fontsize)
        plt.title("Contour Plot", fontsize = fontsize)

        plt.tight_layout()

        plt.show()

###
#   Copula generators for Archimedean (Frank and Clayton) copulas. Allows for easy and accurate copula generation
#   at any dimension in terms of univariate functions (generator and inverse generator).
###

def ClaGen(x, t = 1): return 1/t * (x **(-t) -1 )       # Clayton copula generator
def ClaInv(x, t = 1): return (1 + t * x) ** (-1/t)      # Inverse generator

def FGen(x, s = 1):                     # Frank generator
    X1 = np.exp( -(x * s) ) -1
    X2 = np.exp( -s ) -1
    return -np.log( X1 / X2)

def FInv(x, s = 1):                     # Inverse
    X1 = np.exp( -x ) * (np.exp(-s) - 1)
    X2 = np.log(1 + X1)
    return - (X2 / s)

###
#   Copula functions and constructors
###
def indep(x,y): return x*y
def perf(x,y): return min(x,y)
def opp(x,y): return max(x+y-1,0)
def Cla(x, y, t = 1): return ClaInv( ClaGen(x, t) + ClaGen(y, t), t)
def F(x,y,s = 1): return FInv( FGen(x, s) + FGen(y, s), s)
def Gau(x,y,r=0):
    if x == 0: return 0
    if y == 0: return 0
    return mvn.cdf([norm.ppf(x), norm.ppf(y)], mean = [0, 0], cov=[[1, r], [r, 1]])



# Copula constructors
def pi(steps = 200):
    x = y = np.linspace(0, 1, num=steps)
    cdf = np.array([[xs * ys for xs in x] for ys in y])
    return Copula(cdf, indep)

def M(steps = 200):
    x = y = np.linspace(0, 1, num=steps)
    cdf = np.array([[perf(xs, ys) for xs in x] for ys in y])
    return Copula(cdf, perf)

def W(steps = 200):
    x = y = np.linspace(0, 1, num=steps)
    cdf = np.array([[opp(xs, ys) for xs in x] for ys in y])
    return Copula(cdf, opp)

def Frank(s = 0, steps = 200):      #   s is real; inf for perfect, 0 for indep, -inf for oposite

    if s == float('-inf'):         # Limit should be set earlier
        C = W()
        return Copula(C.cdf, F, float('-inf'))
    if s == 0:
        C = pi()
        return Copula(C.cdf, F, 1)
    if s == float('inf'):
        C = M()
        return Copula(C.cdf, F, float('inf'))

    x = y = np.linspace(0, 1, num=steps)
    cdf = np.array([[F(xs, ys, s) for xs in x] for ys in y])
    return Copula(cdf, F, s)

def Clayton(t = 1, steps = 200):    #   t>-1; -1 for opposite, 0 for indep and inf for perfect

    if t == 0:
        C = pi()
        return Copula(C.cdf, Cla , 0)
    if t == -1:
        C = W()
        return Copula(C.cdf, Cla ,-1)
    if t == float('inf'):                       # Limit should be set earlier
        C = M()
        return Copula(C.cdf, Cla , float('inf'))

    x = np.linspace(0, 1, num=steps)
    xx, yy = np.meshgrid(x,x,indexing='ij')

    X = np.array([xx.flatten(), yy.flatten()])
    cdf = Cla(X[0, ], X[1, ], t)
    cdf = cdf.reshape(steps, steps)

    cdf[0,] = 0; cdf[:,0] = 0           # Grounds C

    return Copula(cdf, Cla, t)

def Gaussian(r = 0, steps = 200):   #   -1 <= r <=1 ; -1 for opposite, 1 for indep and 1 for perfect

    if r == 0:
        C = pi()
        return Copula(C.cdf, Gau, 0)
    if r == -1:
        C = W()
        return Copula(C.cdf, Gau, -1)
    if r == 1:
        C = M()
        return Copula(C.cdf, Gau, 1)

    x = np.linspace(0, 1, num=steps)
    xx, yy = np.meshgrid(x,x,indexing='ij')

    X = np.array([xx.flatten(), yy.flatten()])
    cdf = mvn.cdf(norm.ppf(X.transpose()), mean=[0,0], cov=[[1, r], [r, 1]])
    cdf = cdf.reshape(steps, steps)

    cdf[0,] = 0; cdf[:,0] = 0           # Grounds C

    return Copula(cdf, Gau, r)
