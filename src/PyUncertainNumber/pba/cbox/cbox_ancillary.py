""" 
ancillary codes for cbox written by Scott which are originally in R
are translated to Python by the author of this snippet.
- mostly can be ignored 
"""


# ---------------------ancillary modules by Scott---------------------#

many = 2000  # increase for more accuracy

def beta(v, w):
    if v == 0:
        return np.zeros(many)
    elif w == 0:
        return np.ones(many)
    else:
        return np.random.beta(v, w, many)

def bernoulli(p):
    return np.random.uniform(0, 1, many) < p

def betabinomial(size, v, w):
    return np.random.binomial(size, beta(v, w))

def negativebinomial(size, prob):
    return np.random.negative_binomial(size, prob, many)

def gamma(shape, rate=1, scale=1):
    rate = 1 / scale
    return np.random.gamma(shape, 1/rate, many)

def gammaexponential(shape, rate=1, scale=1):
    rate = 1 / scale
    return np.random.exponential(gamma(shape, rate), many)

def normal(m, s):
    return np.random.normal(m, s, many)

def student(v):
    return np.random.standard_t(v, many)

def chisquared(v):
    return np.random.chisquare(v, many)

def lognormal(m, s):
    return np.random.lognormal(m, s, many)

def uniform(a, b):
    return np.random.uniform(a, b, many)

def histogram(x):
    return x[np.trunc(np.random.uniform(0, 1, many) * len(x)).astype(int)]

def mixture(x, w):
    r = np.sort(np.random.uniform(0, 1, many))
    x = np.concatenate(([x[0]], x))
    w = np.cumsum(np.concatenate(([0], w))) / np.sum(w)
    u = []
    j = len(x) - 1
    for p in reversed(r):
        while True:
            if w[j] <= p:
                break
            j -= 1
        u.append(x[j])
    return np.array(u)[np.random.permutation(len(u))]
    
def env(x, y):
    return np.concatenate((x, y))

import numpy as np
from scipy.special import beta

Bzero = 1e-6
Bone = 1 - Bzero

def ratiokm(k, m):
    return 1 / (1 + m / k)

def ratioKN(k, n):
    return k / n

def jeffkm(k, m):
    return beta(k + 0.5, m + 0.5)

def jeffKN(k, n):
    return beta(k + 0.5, n - k + 0.5)

def km(k, m):
    if k < 0 or m < 0:
        raise ValueError('Improper arguments to function km')
    return np.clip(beta(k, m + 1), Bzero, Bone)

def KN(k, n):
    if k < 0 or n < k:
        raise ValueError('Improper arguments to function KN')
    return np.clip(beta(k, n - k + 1), Bzero, Bone)

def Bppv(p, s, t):
    return 1 / (1 + ((1 / p - 1) * (1 - t)) / s)

def Bnpv(p, s, t):
    return 1 / (1 + (1 - s) / (t * (1 / p - 1)))

def ppv(pk, pm, sk, sm, tk, tm, mk=km):
    return Bppv(mk(pk, pm), mk(sk, sm), mk(tk, tm))

def npv(pk, pm, sk, sm, tk, tm, mk=km):
    return Bnpv(mk(pk, pm), mk(sk, sm), mk(tk, tm))

import numpy as np

def ANDi(x, y):
    nx = len(x)
    ny = len(y)
    if (nx == 1) and (ny == 1):
        return x * y
    return np.concatenate((x[:many] * y[:many], x[many:nx] * y[many:ny]))

def ORi(x, y):
    nx = len(x)
    ny = len(y)
    if (nx == 1) and (ny == 1):
        return 1 - (1 - x) * (1 - y)
    return np.concatenate((1 - (1 - x[:many]) * (1 - y[:many]), 1 - (1 - x[many:nx]) * (1 - y[many:ny])))

def OPi(x, y, op):
    nx = len(x)
    ny = len(y)
    if op == '-':
        return OPi(x, np.concatenate((-y[many:ny], -y[:many])), '+')
    if op == '/':
        return OPi(x, np.concatenate((1 / y[many:ny], 1 / y[:many])), '*')
    if (nx == 1) and (ny == 1):
        return eval(f"{x} {op} {y}")
    if nx == 1:
        return np.concatenate((eval(f"{x} {op} {y[:many]}"), eval(f"{x} {op} {y[many:ny]}")))
    if ny == 1:
        return np.concatenate((eval(f"{x[:many]} {op} {y}"), eval(f"{x[many:nx]} {op} {y}")))
    return np.concatenate((eval(f"{x[:many]} {op} {y[:many]}"), eval(f"{x[many:nx]} {op} {y[many:ny]}")))

def opi(x, y, op):
    nx = len(x)
    ny = len(y)
    if (nx == 1) and (ny == 1):
        return eval(f"{x} {op} {y}")
    if nx == 1:
        return opi(np.repeat(x, 2 * many), y)
    if ny == 1:
        return opi(x, np.repeat(y, 2 * many))
    if op == '+':
        return np.concatenate((x[:many] + y[:many], x[many:nx] + y[many:ny]))
    if op == '-':
        return opi(x, np.concatenate((-y[many:ny], -y[:many])), '+')
    if op == '*':
        return np.concatenate((x[:many] * y[:many], x[many:nx] * y[many:ny]))
    if op == '/':
        return opi(x, np.concatenate((1 / y[many:ny], 1 / y[:many])), '*')
    if op == '^':
        return np.concatenate((x[:many] ** y[:many], x[many:nx] ** y[many:ny]))
    if op in ['min', 'pmin']:
        return np.concatenate((np.minimum(x[:many], y[:many]), np.minimum(x[many:nx], y[many:ny])))
    if op in ['max', 'pmax']:
        return np.concatenate((np.maximum(x[:many], y[:many]), np.maximum(x[many:nx], y[many:ny])))
    raise ValueError('ERROR unknown operator in opi')

import numpy as np
import matplotlib.pyplot as plt

def plotbox(b, new=True, col='blue', lwd=2, xlim=None, ylim=(0, 1), xlab='', ylab='Prob', many=100):
    def edf(x, col, lwd):
        n = len(x)
        s = np.sort(x)
        plt.plot([s[0], s[0]], [0, 1/n], lw=lwd, color=col)
        for i in range(1, n):
            plt.plot([s[i-1], s[i], s[i]], [i-1, i-1, i/n], color=col, lw=lwd)

    b = np.where(b == -np.inf, xlim[0] - 10, b)
    b = np.where(b == np.inf, xlim[1] + 10, b)
    
    if new:
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
    
    if len(b) < many:
        edf(b, col, lwd)
    else:
        edf(
            np.concatenate((([np.min(b)], [np.max(b)], b[:min(len(b), many)])), axis=None), col, lwd)
    
    if many < len(b):
        edf(np.concatenate(([np.min(b)], [np.max(b)], b[many:]), axis=None), col, lwd)

# def ci(b, c=0.95, alpha=(1-c)/2, beta=1-(1-c)/2, many=100):
#     left = np.sort(b[:many])[int(round(alpha * many))]
#     if many < len(b):
#         right = np.sort(b[many:])[int(round(beta * many))]
#     else:
#         right = np.sort(b[:many])[int(round(beta * many))]
#     return np.array([left, right])

def Get_Q(m_in, c_in, k=None):
    if k is None:
        k = np.arange(0, m_in * c_in + 1)

    def Q_size_GLBL(m):
        return 1 + m + m * (m + 1) / 2 + m * (m + 1) * (m + 2) * (3 * m + 1) / 24

    def Q_size_LoCL(m, c):
        return 1 + c + m * c * (c + 1) / 2

    def Grb_Q(m_in, c_in, Q_list):
        m = max(m_in, c_in)
        c = min(m_in, c_in)
        i_min = Q_size_GLBL(m - 1) + Q_size_LoCL(m, c - 1) + 1
        return Q_list[i_min:i_min + m * c + 1]

    def AddingQ(m, Q_list):
        Q_list[Q_size_GLBL(m - 1)] = 1
        for c in range(1, m + 1):
            i_min = Q_size_GLBL(m - 1) + Q_size_LoCL(m, c - 1) + 1
            Q1 = np.concatenate((Grb_Q(m - 1, c, Q_list), np.zeros(c)), axis=None)
            Q2 = np.concatenate((np.zeros(m), Grb_Q(m, c - 1, Q_list)), axis=None)
            Q_list[i_min:i_min + m * c + 1] = Q1 + Q2
        return Q_list[Q_size_GLBL(m - 1):Q_size_GLBL(m) + 1]

    def Bld_Q(m_top):
        Q_out = np.zeros(Q_size_GLBL(m_top))
        Q_out[0] = 1
        for m in range(1, m_top + 1):
            Q_out[Q_size_GLBL(m - 1):Q_size_GLBL(m)] = AddingQ(m, Q_out)
        return Q_out

    m = max(m_in, c_in)
    c = min(m_in, c_in)
    return Grb_Q(m, c, Bld_Q(m))[k]

