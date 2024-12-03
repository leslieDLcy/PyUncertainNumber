import numpy as np
import matplotlib.pyplot as plt
from PyUncertainNumber.UC.stats import *

# ---------------------Test---------------------#
''' method of matching moments (MoM) constructors '''

print('****** 13')
many = 2000


def sh(x, t, Data=None):
    if Data is None:
        Data = data
    plt.title(t)
    edf(Data, 'y')
    edf(x)
    plt.show()


def ecdf(d):
    d = np.array(d)
    N = d.size
    pp = np.concatenate((np.arange(N), np.arange(1, N+1)))/N
    dd = np.concatenate((d, d))
    dd.sort()
    pp.sort()
    return dd, pp


def edf(d, c=None, lw=None, ls=None):
    if d.size == (2*many):  # p-box
        z, p = ecdf(d[0:many])
        plt.plot(z, p, c=c, lw=lw, ls=ls)
        z, p = ecdf(d[(many):(2*many)])
        plt.plot(z, p, c=c, lw=lw, ls=ls)
    else:  # distribution
        z, p = ecdf(d)
        plt.plot(z, p, c=c, lw=lw, ls=ls)
    # plt.ylabel('Cumulative probability') # just makes the graph smaller


data = np.random.uniform(size=25)
datat = 2 * np.random.normal(size=25)
data2 = 2 * data
data10 = np.round(data*10)
datap = data + 1
data10p = np.round(data*10) + 1
data100 = 1+np.round(data * 100)
dataBB = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 24,
                  104, 286, 670, 1033, 1343, 1112, 829, 478, 181, 45, 7))
N = 12

x = MMbernoulli(data)
sh(x, 'MMbernoulli(data)')
x = MMbeta(data)
sh(x, 'MMbeta(data)')
x = MMbetabinomial(N, dataBB)
sh(x, 'MMbetabinomial(12,dataBB)', dataBB)
x = MMbinomial(data10p)
sh(x, 'MMbinomial(data100)', data10)
x = MMchisquared(data)
sh(x, 'MMchisquared(data)')
x = MMexponential(data)
sh(x, 'MMexponential(data)')
x = MMF(data2)
sh(x, 'MMF(data2)', data2)
x = MMgamma(data10)
sh(x, 'MMgamma(data10)', data10)
x = MMgaussian(data)
sh(x, 'MMgaussian(data)')
x = MMgeometric(data10p)
sh(x, 'MMgeometric(data10p)', data10p)
x = MMpascal(data10p)
sh(x, 'MMpascal(data10p)', data10p)
x = MMgumbel(data)
sh(x, 'MMgumbel(data)')
x = MMextremevalue(data)
sh(x, 'MMextremevalue(data)')
x = MMlognormal(data)
sh(x, 'MMlognormal(data)')
x = MMlaplace(data)
sh(x, 'MMlaplace(data)')
x = MMdoubleexponential(data)
sh(x, 'MMdoubleexponential(data)')
x = MMlogistic(data)
sh(x, 'MMlogistic(data)')
x = MMloguniform(data)
sh(x, 'MMloguniform(data)')
x = MMnormal(data)
sh(x, 'MMlognormal(data)')
x = MMpareto(data)
sh(x, 'MMpareto(data)')
x = MMpoisson(data)
sh(x, 'MMpoisson(data)')
x = MMpowerfunction(data)
sh(x, 'MMpowerfunction(data)')
x = MMt(datat)
sh(x, 'MMt(datat)', datat)
x = MMstudent(datat)
sh(x, 'MMstudent(datat)', datat)
x = MMuniform(data)
sh(x, 'MMuniform(data)')
x = MMrectangular(data)
sh(x, 'MMrectangular(data)')
x = MMtriangular(data)
sh(x, 'MMtriangular(data)')
