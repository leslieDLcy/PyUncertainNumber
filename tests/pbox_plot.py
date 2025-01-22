import matplotlib.pyplot as plt
from PyUncertainNumber import pba

### pba level ###
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 7), layout="constrained")
# first row
pba.normal([0, 1], [2, 3]).display(style="band", ax=axs[0, 0], title="N([0,1],[2,3])")
pba.uniform([0, 1], 3).display(style="band", ax=axs[0, 1], title="U([0,1], 3)")
pba.beta([0.7, 1], [3, 4]).display(
    style="band", ax=axs[0, 2], title="Beta([0.7,1],[3,4])"
)
pba.gamma([5, 6], 2).display(style="band", ax=axs[0, 3], title="Gamma([5,6],2)")

# second row
pba.lognormal([2, 3], [1, 5]).display(
    style="band", ax=axs[1, 0], title="Lognormal([2,3],[1,5])"
)
pba.expon([0.4, 0.6]).display(style="band", ax=axs[1, 1], title="Exp([0.4, 0.6])")
pba.chi2([20, 50]).display(style="band", ax=axs[1, 2], title="Chisquared([20, 50])")
pba.cauchy([1, 100], 1).display(style="band", ax=axs[1, 3], title="Cauchy([1,100], 1)")
plt.show()
