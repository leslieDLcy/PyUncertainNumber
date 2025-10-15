import numpy as np
import scipy.stats as sps
from pyuncertainnumber import make_vec_interval, Interval


def cauchy_deviate_method(
    func,
    input_vector_interval: list[Interval],
    n_sam: int = 200,
) -> Interval:
    """Cauchy Deviate Method for interval propagation

    args:
        func: The function, vectorised style, to be evaluated.
        input_vector_interval: The input vector of intervals.
        n_sam: The number of samples to draw from each Cauchy distribution.

    note:
        The function must be vectorised, i.e. it must be able to take in a
        2D array of shape (n, d) and return a 1D array of shape (n,).
    """
    # enforce vec-Interval
    input_vector_interval = make_vec_interval(input_vector_interval)

    # of dimension d
    nominal_measurement = input_vector_interval.mid

    # of dimension d
    scalar_param = input_vector_interval.rad
    mean_y, y_scalar_param = cauchy_deviate_raw(
        func, nominal_measurement, scalar_param, n_sam
    )
    return Interval.from_meanform(mean_y, y_scalar_param)


def cauchy_deviate_raw(
    func, nominal_measurement: np.ndarray, scalar_param: np.ndarray, n_sam: int = 200
) -> tuple:
    """Raw implementation of the Cauchy Deviate Method

    Args:
        func: The function, vectorised style, to be evaluated.
        nominal_measurement: The nominal measurement values, i.e. x tilda.
        scalar_param: The scalar parameters for each component Cauchy distribution, i.e. Delta.
        n_sam: The number of samples to draw from each Cauchy distribution.

    Returns:
        mean_y and Delta_y
    """

    # of dimension d
    nominal_measurement = np.array(nominal_measurement)

    # of dimension d
    scalar_param = np.array(scalar_param)
    samples = sps.cauchy.rvs(
        scale=scalar_param[:, None], size=(len(scalar_param), n_sam)
    ).T  # samples of shape (n_sam, d)

    # find the max of each row
    K = np.max(samples, axis=1)

    normalised_ = samples / K[:, None]
    semi_actual_inputs = nominal_measurement - normalised_

    # vec
    f_mid = func(nominal_measurement[np.newaxis, :])
    f_semi = func(semi_actual_inputs)
    y_diffs = f_mid - f_semi

    # normalised output difference
    normalised_y_diff = y_diffs * K

    outout_scalar_param = sps.cauchy.fit(normalised_y_diff, floc=0)

    return f_mid.item(), outout_scalar_param[1].item()
