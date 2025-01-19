import pyuncertainnumber as pun
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pyuncertainnumber import pba

# *  ---------------------construction---------------------*#
with mpl.rc_context({"text.usetex": True}):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 7), layout="constrained")

    # first row
    pun.normal([0, 1], [2, 3]).display(
        style="band", ax=axs[0, 0], title="N([0,1],[2,3])"
    )
    pun.uniform([0, 1], 3).display(style="band", ax=axs[0, 1], title="U([0,1], 3)")
    pun.beta([0.7, 1], [3, 4]).display(
        style="band", ax=axs[0, 2], title="Beta([0.7,1],[3,4])"
    )
    pun.gamma([5, 6], 2).display(style="band", ax=axs[0, 3], title="Gamma([5,6],2)")

    # second row
    pun.lognormal([2, 3], [1, 5]).display(
        style="band", ax=axs[1, 0], title="Lognormal([2,3],[1,5])"
    )
    pun.expon([0.4, 0.6]).display(style="band", ax=axs[1, 1], title="Exp([0.4, 0.6])")
    pun.chi2([20, 50]).display(style="band", ax=axs[1, 2], title="Chisquared([20, 50])")
    pun.cauchy([1, 100], 1).display(
        style="band", ax=axs[1, 3], title="Cauchy([1,100], 1)"
    )
    # plt.show()

# *  ---------------------aggregation---------------------*#

from pyuncertainnumber import stochastic_mixture

lower_endpoints = np.random.uniform(-0.5, 0.5, 7)
upper_endpoints = np.random.uniform(0.5, 1.5, 7)
m_weights = [0.1, 0.1, 0.25, 0.15, 0.1, 0.1, 0.2]
# a list of nInterval objects
nI = [pba.I(couple) for couple in zip(lower_endpoints, upper_endpoints)]
pbox_mix = stochastic_mixture(nI, weights=m_weights, display=True, return_type="pbox")
print("the result of the mixture operation")
print(pbox_mix)

# *  ---------------------arithmetic---------------------*#
# an interval
# an interval
a = pba.I([2, 3])
# _ = a.display(style="band", title="Interval [2,3]")
# a precise distribution
b = pun.norm(0, 1)
# _ = b.display(title="$N(0, 1)$")
t = a + b
print(t)
