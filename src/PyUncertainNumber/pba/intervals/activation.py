from .number import Interval
from .methods import exp
import numpy as np


""" a series of popular activation functions for interval arithmetic """


def sigmoid(x: Interval):
    return 1 / (1 + exp(-x))


def tanh(x: Interval):
    return (exp(2 * x) - 1) / (exp(2 * x) + 1)


def relu(x):
    xs = x.shape
    x0 = np.zeros(xs)
    return np.maximum(x0, x)
