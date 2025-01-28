import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from pyuncertainnumber import UncertainNumber


def outward_direction(x: list,
                      n_disc: Union[int, np.ndarray] = 10,
                      tOp: Union[float, np.ndarray] = 0.999,
                      bOt: Union[float, np.ndarray] = 0.001
                      ):  # Specify return type
    """
    args:
        x (list): A list of UncertainNumber objects.
        f (Callable): The function to evaluate.
        results (dict): A dictionary to store the results (optional).
        method (str): The method which will estimate bounds of each combination of focal elements (default is the endpoint)
        lim_Q (np.array): Quantile limits for discretization.
        n_disc (int): Number of discretization points.

    signature:
        second_order_propagation_method(x: list, f: Callable, results: dict, method: str, lim_Q: np.array, n_disc: int) -> dict

    notes:
        Performs second-order uncertainty propagation for mixed uncertain numbers 

    returns:
        dict: A dictionary containing the results
    """
    d = len(x)  # dimension of uncertain numbers
    bounds_x = []
    ranges = np.zeros((2, d))
    u_list = []  # List to store 'u' for each uncertain number

    for i, un in enumerate(x):
        print(f"Processing variable {i + 1} with essence: {un.essence}")

        if un.essence == "distribution":
            # perform outward directed discretisation
            if isinstance(n_disc, int):
                nd = n_disc + 1
            else:
                nd = n_disc[i] + 1  # Use the i-th element of n_disc array

            distribution_family = un.distribution_parameters[0]

            if distribution_family == 'triang':
                u = np.linspace(0, 1, nd)
            else:
                if isinstance(tOp, np.ndarray):
                    top = tOp[i]
                else:
                    top = tOp
                if isinstance(bOt, np.ndarray):
                    bot = bOt[i]
                else:
                    bot = bOt
                u = np.linspace(bot, top, nd)  # Use bOt and tOp here

        elif un.essence == "pbox":
            distribution_family = un.distribution_parameters[0]
            if distribution_family == 'triang':
                if isinstance(n_disc, int):
                    nd = n_disc + 1
                else:
                    nd = n_disc[i] + 1  # Use the i-th element of n_disc array
                u = np.linspace(0, 1, nd)
            else:
                if isinstance(n_disc, int):
                    nd = n_disc + 1
                else:
                    nd = n_disc[i] + 1  # Use the i-th element of n_disc array

                if isinstance(tOp, np.ndarray):
                    top = tOp[i]
                else:
                    top = tOp
                if isinstance(bOt, np.ndarray):
                    bot = bOt[i]
                else:
                    bot = bOt
                u = np.linspace(bot, top, nd)  # Use bOt and tOp here

        else:  # un.essence == "interval"
            u = np.array([0.0, 1.0])  # Or adjust as needed for intervals

        u_list.append(u)  # Add 'u' to the list

        # Generate discrete p-boxes
        match un.essence:
            case "distribution":
                # Calculate xl and xr for distributions (adjust as needed)
                # Assuming un.ppf(u) returns a list or array
                temp_xl = un.ppf(u_list[i][:-1])
                # Adjust based on your distribution
                temp_xr = un.ppf(u_list[i][1:])
                # Create a 2D array of bounds
                rang = np.array([temp_xl, temp_xr]).T
                bounds_x.append(rang)
                ranges[:, i] = np.array([temp_xl[0], temp_xr[-1]])

            case "interval":
                # Repeat lower bound for all quantiles
                temp_xl = np.array([un.bounds[0]])
                # Repeat upper bound for all quantiles
                temp_xr = np.array([un.bounds[1]])
                # Create a 2D array of bounds
                rang = np.array([temp_xl, temp_xr]).T
                bounds_x.append(rang)
                ranges[:, i] = un.bounds  # np.array([un.bounds])

            case "pbox":
                temp_xl = un.ppf(u_list[i][:-1])[0]
                temp_xr = un.ppf(u_list[i][1:])[1]
                # Create a 2D array of bounds
                rang = np.array([temp_xl, temp_xr]).T
                bounds_x.append(rang)
                ranges[:, i] = np.array([temp_xl[0], temp_xr[-1]])

            case _:
                raise ValueError(f"Unsupported uncertainty type: {un.essence}")

    return temp_xl, temp_xr


def plotPbox_pbox(xL, xR,  p=None):
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

    # # Add the normal distribution
    # mean = 1
    # std = 0.1
    # x_norm = np.linspace(mean - 3.5 * std, mean + 3.5 * std, 100)  # Generate x-values for the normal distribution
    # y_norm = norm.cdf(x_norm, mean, std)  # Calculate the corresponding PDF values
    # plt.plot(x_norm, y_norm, color='blue')  # Plot the normal distribution

    # Add the pbox distribution
    y = UncertainNumber(essence='pbox', distribution_parameters=[
                        "gaussian", [[1.5, 2.5], 0.20]])

    x_l = y._construct.left  # Generate x-values for the normal distribution
    x_r = y._construct.right  # Generate x-values for the normal distribution
    # Calculate the corresponding PDF values
    y_val = np.linspace(0, 1, len(y._construct.left))
    plt.plot(x_l, y_val, color='blue')  # Plot the normal distribution
    plt.plot(x_r, y_val, color='green')  # Plot the normal distribution

    # Add x and y axis labels
    plt.xlabel("U", fontsize=14)
    plt.ylabel("Cumulative Probability", fontsize=14)
    # Increase font size for axis numbers
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # plt.legend()
    # plt.show()
    # # Highlight the points (xL, p)
    # plt.scatter(xL, p, color=colors, marker='o', edgecolors='black', zorder=3)

    # # Highlight the points (xR, p)
    # plt.scatter(xR, p, color=colors, marker='o', edgecolors='black', zorder=3)

    # plt.fill_betweenx(p, xL, xR, color=colors, alpha=0.5)
    # plt.plot( [xL[0], xR[0]], [0, 0],color=colors, linewidth=3)
    # plt.plot([xL[-1], xR[-1]],[1, 1],  color=colors, linewidth=3)
    plt.show()


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


means = np.array([1, 2, 3, 4, 5])
stds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

n_disc = 10  # Number of discretizations

x = [
    UncertainNumber(essence='distribution', distribution_parameters=[
                    "gaussian", [means[0], stds[0]]])

    #  UncertainNumber(essence = 'interval', bounds= [means[1]-2* stds[1], means[1]+2* stds[1]]),
    # UncertainNumber(essence = 'interval', bounds= [means[2]-2* stds[2], means[2]+2* stds[2]])
]

xl, xr = outward_direction(x)

# plotPbox(xl, xr, p=None)
# plt.show()


x0 = [
    UncertainNumber(essence='pbox', distribution_parameters=[
                    "gaussian", [[1.5, 2.5], 0.20]])
    # UncertainNumber(essence = 'interval', bounds = [2.5, 3.5])

]

y = UncertainNumber(essence='pbox', distribution_parameters=[
                    "gaussian", [[0.5, 1.5], 0.10]])
print('y', y._construct.left)
xl, xr = outward_direction(x0)
plotPbox_pbox(xl, xr, p=None)
plt.show()


def plot_interval(lower_bound, upper_bound, color='blue', label=None):
    """
    Plots an interval on a Matplotlib plot.

    Args:
        lower_bound (float): The lower bound of the interval.
        upper_bound (float): The upper bound of the interval.
        color (str, optional): The color of the interval line. Defaults to 'blue'.
        label (str, optional): The label for the interval in the legend. Defaults to None.
    """

    plt.plot([lower_bound, upper_bound], [1, 1], color=color,
             linewidth=2, label=label)  # Plot the interval line
    plt.plot([lower_bound, lower_bound], [0, 1], color=color,
             linewidth=2, label=label)  # Plot the interval line
    plt.plot([upper_bound, upper_bound], [0, 1], color=color,
             linewidth=2, label=label)  # Plot the interval line

    # Add x and y axis labels
    plt.xlabel("V", fontsize=14)
    plt.ylabel("Cumulative Probability", fontsize=14)
    # Increase font size for axis numbers
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)


y = UncertainNumber(essence='interval', bounds=[2.5, 3.5])
plot_interval(2.5, 3.5)
plt.show()
