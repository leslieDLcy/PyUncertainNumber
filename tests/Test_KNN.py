import numpy as np
import matplotlib.pyplot as plt
from pyuncertainnumber.calibration.knn import KNNCalibrator

rng = np.random.default_rng(12)


def paraboloid_model(theta, xi=0.0, A=1.0, B=0.5, C=1.5):
    """Vectorized paraboloid, mild noise; supports scalar or vector xi."""
    theta = np.atleast_2d(theta).astype(float)
    x1, x2 = theta[:, 0], theta[:, 1]
    xi = np.asarray(xi, float)
    if xi.ndim == 0:
        xi = np.full(theta.shape[0], xi)
    elif xi.ndim == 2:
        xi = xi.ravel()
    y = A * x1**2 + B * x1 * x2 * (1.0 + xi) + C * (x2 + xi) ** 2
    y = y + 0.2 * np.random.randn(theta.shape[0])  # small noise
    return y.reshape(-1, 1) if theta.shape[0] > 1 else np.array([y.item()])


def theta_sampler(n, lb=-15, ub=15):
    return np.random.uniform(lb, ub, size=(n, 2))


def scatter_post(ax, theta, truth=None, title="", alpha=0.30, s=6, label="Posterior"):
    ax.scatter(theta[:, 0], theta[:, 1], s=s, alpha=alpha, label=label)
    if truth is not None:
        ax.scatter(
            truth[:, 0], truth[:, 1], c="r", marker="x", s=60, label="θ true cloud"
        )
    ax.set_title(title)
    ax.set_xlabel("θ1")
    ax.set_ylabel("θ2")
    ax.grid(True)
    ax.legend()


def test_knn_single_design():
    """Test KNN calibration with single design point and generate visualization."""
    # Case 1 - 1 sample (y), 1 target (θ point-valued), 1 experiment (ξ)
    theta_true_c1 = rng.normal(0, 4, size=(1, 2))  # unknown
    xi_list_c1 = [0.0]  # selected (experiment)
    y_emp = paraboloid_model(theta_true_c1, xi_list_c1)
    observations_c1 = [(y_emp, xi_list_c1)]
    print(
        f"CASE 1 - designs: {len(observations_c1)} samples: {observations_c1[0][0].shape[0]}"
    )

    # set up knn calibrator
    calib_single = KNNCalibrator(knn=500, evaluate_model=True)
    calib_single.setup(
        model=paraboloid_model,
        theta_sampler=theta_sampler,
        xi_list=xi_list_c1,
        n_samples=500_000,
    )
    # run calibration
    post_single = calib_single.calibrate(
        observations=observations_c1, combine="stack"
    )  # pooled kNN
    theta_post_single = post_single["theta"]

    # Basic assertions
    assert theta_post_single is not None
    assert theta_post_single.shape[1] == 2  # 2D parameter space
    assert len(theta_post_single) > 0

    # visualize
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter_post(
        ax,
        theta_post_single,
        truth=np.atleast_2d(theta_true_c1),
        title=f"Unified kNN — single design (ξ*={xi_list_c1}, pooled neighbors)",
    )
    # plt.show()
