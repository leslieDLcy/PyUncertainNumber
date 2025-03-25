from pyuncertainnumber import UncertainNumber
from pyuncertainnumber.propagation.mixed_uncertainty.varied_discretisation_propagation import varied_discretisation_propagation_method 
import numpy as np
import matplotlib.pyplot as plt

def plotPbox(xL, xR, p=None):
    """
    Plots a p-box (probability box) using matplotlib.

    Args:
        xL (np.ndarray): A 1D NumPy array of lower bounds.
        xR (np.ndarray): A 1D NumPy array of upper bounds.
        p (np.ndarray, optional): A 1D NumPy array of probabilities corresponding to the intervals.
                                   Defaults to None, which generates equally spaced probabilities.
        color (str, optional): The color of the plot. Defaults to 'k' (black).
    """
    xL = np.squeeze(xL)  # Ensure xL is a 1D array
    xR = np.squeeze(xR)  # Ensure xR is a 1D array

    if p is None:
        # p should have one more element than xL/xR
        p = np.linspace(0, 1, len(xL) + 1)

    # Plot the step functions
    plt.step(np.concatenate(([xL[0]], xL)), p, where='post', color='black')
    plt.step(np.concatenate(([xR[0]], xR)), p, where='post', color='red')

    # Add bottom and top lines to close the box
    plt.plot([xL[0], xR[0]], [0, 0], color='red')  # Bottom line
    plt.plot([xL[-1], xR[-1]], [1, 1], color='black')  # Top line

    # Add x and y axis labels
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Cumulative Probability", fontsize=14)
    # Increase font size for axis numbers
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()

def Fun(x):
    input1 = x[0]
    input2 = x[1]
    input3 = x[2]
    input4 = x[3]
    input5 = x[4]

    output1 = input1 + input2 + input3 + input4 + input5
    output2 = input1 * input2 * input3 * input4 * input5

    return np.array([output1]) #, output2

means = np.array([1, 2, 3, 4, 5])
stds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

x = [
    UncertainNumber(essence='distribution',
                    distribution_parameters=["gaussian", [1, 0.1]]),
    UncertainNumber(essence='distribution',
                    distribution_parameters=["gaussian", [2, 0.2]]),
    UncertainNumber(essence='distribution',
                    distribution_parameters=["gaussian", [3, 0.3]]),
    UncertainNumber(essence='distribution',
                    distribution_parameters=["gaussian", [4, 0.4]]),
    UncertainNumber(essence='distribution',
                    distribution_parameters=["gaussian", [5, 0.5]])
]

results = varied_discretisation_propagation_method(x=x, f=Fun, method = 'varied_discretisation_genetic_opt', n_disc=5)
print(results.raw_data['min'][0]['f'])
print(results.raw_data['max'][0]['f'])
print(results.raw_data['min'][0]['f'])
print(results.raw_data['bounds'])
plotPbox(results.raw_data['min'][0]['f'], results.raw_data['max'][0]['f'], p=None)