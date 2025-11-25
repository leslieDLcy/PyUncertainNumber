# ----------------------------
# Example usage
import numpy as np
import matplotlib.pyplot as plt
from pyuncertainnumber.calibration.knn import KNNCalibrator


# --- import your unified calibrator ---
# from knn import KNNCalibrator
# (Assuming KNNCalibrator is defined in the current file for this snippet.)


def paraboloid_model(theta, xi=0.0, A=1.0, B=0.5, C=1.5):
    """Vectorized paraboloid, mild noise; supports scalar or vector xi."""
    theta = np.atleast_2d(theta).astype(float)
    x1, x2 = theta[:, 0], theta[:, 1]
    xi = np.asarray(xi, float)
    if xi.ndim == 0:
        xi = np.full(theta.shape[0], xi)
    elif xi.ndim == 2:  # if passed as (n,1)
        xi = xi.ravel()
    y = A * x1**2 + B * x1 * x2 * (1.0 + xi) + C * (x2 + xi) ** 2
    y = y + 0.2 * np.random.randn(theta.shape[0])  # small noise
    return y.reshape(-1, 1) if theta.shape[0] > 1 else np.array([y.item()])


def theta_sampler(n, lb=-15, ub=15):
    return np.random.uniform(lb, ub, size=(n, 2))


# --------------- Build observations (unknown process) ---------------
N_emp = 100
rng = np.random.default_rng(7)
theta_target = rng.normal(3.1, 0.3, size=(N_emp, 2))  # this is unknown in practice
experiment_designs = [-1.0, 0.0, 1.0, 3.0]

observations = []
for xi in experiment_designs:
    y_emp = paraboloid_model(theta=theta_target, xi=xi)  # shape (100,1) per design
    observations.append((y_emp, xi))

# --------------- kNN calibration multiple designs ---------------
# Use model+sampler so each design gets its own per-design kNN built on the SAME theta grid
calib_joint = KNNCalibrator(knn=100, evaluate_model=True)
calib_joint.setup(
    model=paraboloid_model,
    theta_sampler=theta_sampler,
    xi_list=experiment_designs,
    n_samples=100_000,
)

#  “intersect”
post_joint = calib_joint.calibrate(
    observations=observations, combine="intersect", resample_n=5000
)
theta_post_joint = post_joint["theta"]  # (5000, 2) resampled
# If you wanted grid + weights instead, call with resample_n=None and use post_joint["theta"], post_joint["weights"].

# --------------- SINGLE-DESIGN calibration at xi=0.0 ---------------
xi_star = 0.0
y_obs_many = next(y for (y, xi) in observations if np.isclose(xi, xi_star))

calib_single = KNNCalibrator(knn=100, evaluate_model=True)
calib_single.setup(
    model=paraboloid_model,
    theta_sampler=lambda n: theta_sampler(n, -15, 15),
    xi_list=[xi_star],
    n_samples=100_000,
)
post_single = calib_single.calibrate(
    [(y_obs_many, xi_star)], combine="stack"
)  # pooled kNN
theta_post_many = post_single["theta"]  # shape: (N_emp*knn, 2)

# --------------- Quick nearest() demo ---------------
# Take a single observed y at xi=1.0 and fetch its 10 nearest θ:
y_one = observations[2][0][0]  # first row at design 1.0
theta_knn_10 = calib_joint.nearest(y_one, xi=1.0, k=10)  # (10,2) θ neighbors

# --------------- PLOTS ---------------
# 1) Joint posterior (resampled) vs true cloud
plt.figure(figsize=(6, 5))
plt.scatter(
    theta_post_joint[:, 0],
    theta_post_joint[:, 1],
    s=6,
    alpha=0.25,
    label="Joint posterior (vote, resampled)",
)
plt.scatter(
    theta_target[:, 0],
    theta_target[:, 1],
    c="r",
    marker="x",
    s=40,
    label=r"θ_true samples (unknown)",
)
plt.title("Unified kNN: joint combine='vote' over multiple designs")
plt.xlabel("θ1")
plt.ylabel("θ2")
plt.legend()
plt.grid(True)
plt.show()

# 2) Single-design posterior (kNN pooled) vs true cloud
plt.figure(figsize=(6, 5))
plt.scatter(
    theta_post_many[:, 0],
    theta_post_many[:, 1],
    s=6,
    alpha=0.35,
    label="Single-design posterior (pooled kNN)",
)
plt.scatter(
    theta_target[:, 0],
    theta_target[:, 1],
    c="r",
    marker="x",
    s=40,
    label=r"θ_true samples (unknown)",
)
plt.title(f"Unified kNN: single-design at ξ*={xi_star}")
plt.xlabel("θ1")
plt.ylabel("θ2")
plt.legend()
plt.grid(True)
plt.show()

# 3) Show the 10 nearest θ for one observed y at xi=1.0 (sanity check)
print("Ten nearest θ for one observed y at ξ=1.0:\n", theta_knn_10)
