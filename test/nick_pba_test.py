
# this was placed in 'pba_parametric' file

import matplotlib.pyplot as plt
import time
from scipy.stats import alpha, chi
from PyUncertainNumber.pba.pbox_parametric import Parametric
from PyUncertainNumber.pba.interval import Interval
from PyUncertainNumber.pba.pbox_parametric import dist, list_parameters

import scipy.stats as sps
import numpy as np

# A = norm(Interval(0,1),1)
# A.pdf(0)

for d in dist:
    print(f"{d} : \n\t {list_parameters(d)}")

B = Parametric(
    "beta", a=Interval(1, 3), b=Interval(2), support=[10, 20], n_subinterval=5
)
print("Expected Value : {}".format(B.expect()))
print("Interval Value : {}".format(B.interval(0.95)))
print("Interval Value : {}".format(B.isf(0.1)))
print("Mean Value : {}".format(B.mean()))
print("Median Value : {}".format(B.median()))
print("Variance Value : {}".format(B.var()))
print("STD Value : {}".format(B.std()))
print("Entropy Value : {}".format(B.entropy()))
print("Support Value : {}".format(B.support()))
print("Stats Value : {}".format(B.stats()))

s = B.rvs(100)

xb = np.linspace(B.get_support()[0], B.get_support()[1], 100)
B_cdf = B.cdf(xb)
L, R = zip(*B_cdf)

plt.plot(xb, L)
plt.plot(xb, R)
plt.show()

B = sps.beta(2, 2)
x = np.linspace(0, 1, 100)
xx = np.linspace(10, 20, 100)
plt.plot(xx, B.cdf(x))
plt.show()

# from pba import parametric
N = Parametric("norm", Interval(3, 5), Interval(1, 4), n_subinterval=5)
N.plot()

N.get_support()

A = alpha(20)
A.plot()
x = np.linspace(0.04, 0.07, 100)
plt.plot(x, sps.alpha(20).cdf(x))
plt.show()

list_parameters("chi")
C = chi([1, 2])
C.plot()
x = np.linspace(0, 5, 100)
plt.plot(x, sps.chi(2).cdf(x))
plt.show()

"""
Speed check 
"""
t0 = time.time()
Xi = np.linspace(-15, 15, 200)
pdf0 = [N.pdf(i) for i in Xi]
t1 = time.time()
dt0 = t1 - t0
print("For loop eval : {}sec".format(dt0))

t2 = time.time()
pdf = N.pdf(Xi)
t3 = time.time()
dt1 = t3 - t2
print("Vector eval : {}sec".format(dt1))

print("Time Saving : {}%".format((dt1 / dt0)))
L, R = zip(*pdf)
plt.plot(Xi, L, c="C{}".format(0))
plt.plot(Xi, R, c="C{}".format(0))
plt.legend()
plt.show()

"""
Sub intervalised check
"""
PDF_n = []
n_check = [0, 16, 20]
for n in n_check:
    N = Parametric("norm", Interval(0, 5), Interval(1, 4), n_subinterval=n)
    PDF_n.append(N.pdf(Xi))

for i in range(len(n_check)):
    L, R = zip(*PDF_n[i])
    plt.plot(Xi, L, c="C{}".format(i), label="sub int - {}".format(n_check[i]))
    plt.plot(Xi, R, c="C{}".format(i))
plt.legend()
plt.show()

"""
Outputs
"""
L, R = zip(*pdf0)
plt.figure()
plt.title("PBox Density Parametric Compute")
plt.plot(Xi, L)
plt.plot(Xi, R)
plt.show()

cdf = N.cdf(Xi)
L, R = zip(*cdf)

plt.figure()
plt.title("PBox Cumulative")
plt.plot(N.left, np.linspace(0, 1, len(N.left)), c="C1", label="PBA")
plt.plot(N.right, np.linspace(0, 1, len(N.left)), c="C1")
plt.plot(Xi, L, c="C0", label="Full dist")
plt.plot(Xi, R, c="C0")
plt.legend()
plt.show()

sf = N.sf(Xi)
L, R = zip(*sf)

plt.figure()
plt.title("PBox Survival Function")
plt.plot(Xi, L)
plt.plot(Xi, R)
plt.show()

logcdf = N.logcdf(Xi)
L, R = zip(*logcdf)

# Is this right?
plt.figure()
plt.title("PBox Log CDF Function")
plt.plot(Xi, L)
plt.plot(Xi, R)
plt.show()

logpdf = N.logpdf(Xi)
L, R = zip(*logpdf)

# Is this right?
plt.figure()
plt.title("PBox Log PDF Function")
plt.plot(Xi, L)
plt.plot(Xi, R)
plt.show()

alpha = np.linspace(0, 0.99, 200)
CI = [N.interval(i) for i in alpha]
L, R = zip(*CI)
L0, L1 = zip(*L)
R0, R1 = zip(*R)
# Is this right?
plt.figure()
plt.title("PBox CI Function")
plt.plot(L0, alpha, c="r")
# plt.plot(L1,alpha,c='r')
# plt.plot(R0,alpha,c='k')
plt.plot(R1, alpha, c="k")
plt.show()

print("Expected Value: {}".format(N.expect(None)))
