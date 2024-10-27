"""
Scott made some example calculations written in R.
Below shows the translation of the R code to Python.
source: https://sites.google.com/site/confidenceboxes/software
"""


import numpy as np
import matplotlib.pyplot as plt

def plotem(t, d, p=None, q=None, r=None):
    dp = np.concatenate((d, p)) if p is not None else d

    if p is not None:
        plotbox(d, color='gray', xlim=(np.min(dp[np.isfinite(dp)]), np.max(dp[np.isfinite(dp)])))
        plotbox(p, new=False)
        plt.title(f'next {t[0]} value')

    if q is not None:
        plotbox(q)
        plt.title(t[1])
        print(f'95% confidence interval for the {t[1]} is [{signif(ci(q), 3)}]')

    if r is not None:
        plotbox(r)
        plt.title(t[2])
        print(f'95% confidence interval for the {t[2]} is [{signif(ci(r), 3)}]')

def plotbox(data, color='blue', xlim=None, new=True):
    if new:
        plt.figure()
    plt.plot(data, color=color)
    if xlim is not None:
        plt.xlim(xlim)
    plt.show()

def ci(data):
    # Placeholder for confidence interval calculation
    return np.percentile(data, [2.5, 97.5])

def signif(x, digits):
    return np.round(x, digits)

