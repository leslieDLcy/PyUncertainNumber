from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

"""
--------------------------
Editted by Leslie Feb 2024 
MIT License
--------------------------
"""

def plot_intervals(x, y_i, **kwargs):
    """ plot intervals vertically 
    
    args:
        x: array-like precise values
            x-axis coordinates
        y_i: array-like Interval objects
            array of intervals
    """
    
    fig, ax = plt.subplots()

    def basic_plot(x, y_i, **kwargs):
        ax.plot([x, x], [y_i.hi, y_i.lo], 'blue', **kwargs)
        if np.any(y_i.lo == y_i.hi):
            sc_x = x[y_i.lo == y_i.hi]
            sc_y = y_i[y_i.lo == y_i.hi].lo
            ax.scatter(sc_x, sc_y, c='blue', **kwargs)
            
    if len(x.shape) > 1:
        for xx, interval in zip(x, y_i):
            basic_plot([xx, xx], [interval.hi, interval.lo])
    else:
        basic_plot(x, y_i)
    return ax


def plot_lower_bound(x, y_i, **kwargs):
    """ plot lower bound of intervals 
    
    args:
        x: array-like
            x-axis coordinates
        y_i: array-like
            array of intervals
    """
    
    fig, ax = plt.subplots()
    ax.scatter(x, y_i.lo, label='lower bound', **kwargs)
    ax.legend()


