from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


class EpistemicFilter:

    def __init__(
        self,
        xe_samples: NDArray,
        discrepancy_scores: NDArray,
        sets_of_discrepancy: list = None,
    ):
        """The EpistemicFilter method to reduce the epistemic uncertainty space based on discrepancy scores.

        args:
            xe_samples (NDArray): Proposed Samples of epistemic parameters, shape (n_samples, n_dimensions).
                Typically samples from a bounded set of some epistemic parameters.

            discrepancy_scores (NDArray): Discrepancy scores between the model simulations and the observation.
                Associated with each xe sample, shape (n_samples,).

            sets_of_discrepancy (list, optional): List of sets of discrepancy scores for multiple datasets.
                Each element should be an NDArray of shape (n_samples,). Defaults to None.

        tip:
            For performance functions that output multiple responses, some aggregation of discrepancy scores may be used.

        """
        self.xe_samples = xe_samples
        self.discrepancy_scores = discrepancy_scores
        self.sets_of_discrepancy = (
            sets_of_discrepancy if sets_of_discrepancy is not None else None
        )

    def filter(self, threshold: float = 0.1):
        """Filter the epistemic samples based on a discrepancy threshold.

        args:
            threshold (float): The discrepancy threshold for filtering data points.

        returns:
            tuple:
                - the filtered xe samples;
                - hull;
                - lower bounds;
                - upper bounds of the bounding box, or (None, None) if unsuccessful.

        """
        return filter_by_discrepancy(
            self.xe_samples, self.discrepancy_scores, threshold
        )

    def plot_hull_with_bounds(
        self,
        filtered_xe,
        hull=None,
        ax=None,
        show=True,
        x_title=None,
        y_title=None,
        hull_alpha=0.25,
    ):
        """Plot the convex hull and bounding box of the epistemic samples.

        args:
            hull (ConvexHull, optional): Precomputed convex hull. If None, it is computed.

            ax (matplotlib Axes, optional): Existing axes to draw on. If None, a new figure/axes is created.

            show (bool, optional): If True, calls plt.show() at the end. Defaults to True.

            x_title (str, optional): Label for the x-axis. Defaults to None.

            y_title (str, optional): Label for the y-axis. Defaults to None.

            hull_alpha (float, optional): Transparency of the hull surface. Defaults to 0.25.

        returns:
            ax (matplotlib Axes): The axes containing the plot.


        .. figure:: /_static/convex_hull.png
            :alt: convex hull with bounds
            :align: center
            :width: 50%

            Convex hull with bounds illustration.
        """
        return plot_convex_hull_with_bounds(
            filtered_xe,
            hull=hull,
            ax=ax,
            show=show,
            x_title=x_title,
            y_title=y_title,
            hull_alpha=hull_alpha,
        )


##### NASA challenge 2025 #####
# dummy query xc

xc_query = [[0.533, 0.666, 0.5], [0.052631579, 0.421052632, 0.631578947]]

n_steps_so_far = len(xc_query)


# * this is probably problem specific
def plot_convex_hulls(xe_samples, max_discrepancy, n_levels, min_level=0.005):
    """Plot the convex hulls of the 2D projections of Xe
    max_discrepancy: array with discrepancy scores of xe samples
    xe: samples of the epistemic params
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))  # 3 rows, 1 column
    MAX_delta, MIN_delta = np.nanmax(max_discrepancy), np.nanmin(max_discrepancy)

    MIN_delta += min_level
    Kid = 0  # for the legend
    # Use a colormap for red scale
    for ax, xe_idx_2plot in zip(
        axs, [[0, 1], [0, 2], [1, 2]]
    ):  # Loop through each pair of dimensions to plot
        # Set up line styles, markers, and size
        markers = ["x", "x", "x", "x", "x", "x", "x", "o"]

        for i, threshold in enumerate(np.linspace(MAX_delta, MIN_delta, n_levels)):
            filtered_points = xe_samples[max_discrepancy < threshold][:, xe_idx_2plot]
            # Plot filtered points with corresponding color
            ax.scatter(
                filtered_points[:, 0],
                filtered_points[:, 1],
                s=5,
                c="r",
                label=f"Thresh < {threshold:.2f}",
                alpha=0.02,
                marker=markers[i % len(markers)],
            )

        for i, threshold in enumerate(np.linspace(MAX_delta, MIN_delta, n_levels)):
            filtered_points = xe_samples[max_discrepancy < threshold][:, xe_idx_2plot]

            if len(filtered_points) > 2:  # Fill the convex hull with a light color
                hull = ConvexHull(filtered_points)
                for simplex in hull.simplices:  # Outline the convex hull
                    ax.plot(
                        filtered_points[simplex, 0],
                        filtered_points[simplex, 1],
                        "b-",
                        alpha=0.9,
                        linewidth=4,
                    )

        ax.set_title(f"Convex Hulls 2D projection of Xe")
        ax.set_xlabel(f"Xe {1+xe_idx_2plot[0]}")
        ax.set_ylabel(f"Xe {1+xe_idx_2plot[1]}")
        ax.grid()
        if Kid == 0:
            ax.legend(loc="upper right")
            Kid += 1
    plt.tight_layout()
    plt.grid()
    plt.show()


# * this is probably problem specific
def plot_convex_hull_3d(xe_samples, max_discrepancy, n_levels: int = 2):
    """Plot 3d convex hull of Xe

    max_discrepancy: array with discrepancy scores of xe samples
    xe: samples of the epistemic params
    """
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot all points
    ax.scatter(
        xe_samples[:, 0],
        xe_samples[:, 1],
        xe_samples[:, 2],
        label="All points",
        alpha=0.1,
    )
    MAX_delta, MIN_delta = np.nanmax(max_discrepancy), np.nanmin(max_discrepancy)

    for threshold in np.linspace(MAX_delta, MIN_delta + 0.05, n_levels):
        filtered_points = xe_samples[max_discrepancy < threshold]
        ax.scatter(
            filtered_points[:, 0],
            filtered_points[:, 1],
            filtered_points[:, 2],
            label=f"Thresh < {threshold:.2f}",
            alpha=0.1 * (1 - MAX_delta),
        )
        if len(filtered_points) > 3:  # ConvexHull in 3D requires at least 4 points
            hull = ConvexHull(filtered_points)
            for simplex in hull.simplices:
                vertices = filtered_points[simplex]
                poly = Poly3DCollection(
                    [vertices], alpha=0.4, color="black", edgecolor="r"
                )
                ax.add_collection3d(poly)

    # Set default view angle
    ax.view_init(elev=25, azim=60)  # Elevation and azimuth
    # Set plot labels and legend
    ax.set_title("3D Convex Hulls for E={Xe: M(Xe,Xc*,:) <eps} Points")
    ax.set_xlabel("Xe1")
    ax.set_ylabel("Xe2")
    ax.set_zlabel("Xe3")
    ax.legend()
    plt.show()


##### for pun #####


# problem agnostic
def filter_by_discrepancy(xe, discrepancy_scores, threshold=0.1):
    """Computes the intersection of convex hull bounding boxes based on a discrepancy threshold.

    args:
        threshold (float): The discrepancy threshold for filtering data points.

    returns:
        tuple: the hull, and Lower and upper bounds of the intersected bounding box, or (None, None) if unsuccessful.
    """
    # Get the absolute path of the directory containing this script

    filtered_xe = xe[discrepancy_scores < threshold]

    if filtered_xe.size == 0:
        raise ValueError(f"No data points remain after filtering.")

    hull = ConvexHull(filtered_xe)
    # return hull, hull.min_bound, hull.max_bound

    # for multiple thresholds/datasets
    # convex_hulls = []
    # boxes = []

    # convex_hulls.append(hull)
    # boxes.append((hull.min_bound, hull.max_bound))

    # if len(boxes) < 2:
    #     print("Error: Not enough valid bounding boxes to compute intersection.")
    #     return None, None

    # # Compute intersection of bounding boxes
    # box_int_lower = np.maximum(boxes[0][0], boxes[1][0])
    # box_int_upper = np.minimum(boxes[0][1], boxes[1][1])

    # return box_int_lower, box_int_upper

    return filtered_xe, hull, hull.min_bound, hull.max_bound


def plot_convex_hull_with_bounds(
    filtered_xe,
    hull=None,
    ax=None,
    show=True,
    x_title=None,
    y_title=None,
    hull_alpha=0.25,
):
    """
    Plot points, their convex hull, and the axis-aligned bounding box.

    Parameters
    ----------
    filtered_xe : array-like, shape (n_samples, n_dims)
        Input points. Supports 2D or 3D.
    hull : scipy.spatial.ConvexHull, optional
        Precomputed convex hull. If None, it is computed.
    ax : matplotlib Axes or 3D Axes, optional
        Existing axes to draw on. If None, a new figure/axes is created.
    show : bool, default True
        If True, calls plt.show() at the end.
    hull_alpha : float, default 0.25
        Transparency of the hull surface.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes or mplot3d.Axes3D
    hull : scipy.spatial.ConvexHull
    min_bound : ndarray
    max_bound : ndarray
    """
    xe = np.asarray(filtered_xe)
    if xe.ndim != 2:
        raise ValueError("xe must be a 2D array of shape (n_samples, n_dims).")

    n_dims = xe.shape[1]
    if n_dims not in (2, 3):
        raise ValueError("Only 2D or 3D data can be plotted.")

    # Compute hull if needed
    if hull is None:
        hull = ConvexHull(xe)

    # SciPy provides these; fall back to data min/max if not present
    min_bound = getattr(hull, "min_bound", xe.min(axis=0))
    max_bound = getattr(hull, "max_bound", xe.max(axis=0))

    # Set up figure/axes
    if ax is None:
        if n_dims == 2:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # -------------------- 2D CASE --------------------
    if n_dims == 2:
        # Scatter points
        ax.scatter(xe[:, 0], xe[:, 1], s=20, color="blue")

        # Convex hull polygon
        verts = xe[hull.vertices]
        verts_closed = np.vstack([verts, verts[0]])  # close polygon

        ax.fill(
            verts_closed[:, 0],
            verts_closed[:, 1],
            edgecolor="r",
            facecolor="r",
            alpha=hull_alpha,
            linewidth=2,
        )

        # Bounding rectangle
        xmin, ymin = min_bound
        xmax, ymax = max_bound
        rect_x = [xmin, xmax, xmax, xmin, xmin]
        rect_y = [ymin, ymin, ymax, ymax, ymin]

        ax.plot(rect_x, rect_y, "--", linewidth=2, color="g")

        # ax.set_aspect("equal", "box")

    # -------------------- 3D CASE --------------------
    else:
        # Scatter points
        ax.scatter(xe[:, 0], xe[:, 1], xe[:, 2], s=20, color="C0", depthshade=True)

        # Convex hull faces
        faces = [xe[simplex] for simplex in hull.simplices]
        poly = Poly3DCollection(faces, alpha=hull_alpha)
        poly.set_edgecolor("r")
        poly.set_facecolor("r")
        ax.add_collection3d(poly)

        # Bounding box (axis-aligned)
        xmin, ymin, zmin = min_bound
        xmax, ymax, zmax = max_bound

        corners = np.array(
            [
                [xmin, ymin, zmin],
                [xmax, ymin, zmin],
                [xmax, ymax, zmin],
                [xmin, ymax, zmin],
                [xmin, ymin, zmax],
                [xmax, ymin, zmax],
                [xmax, ymax, zmax],
                [xmin, ymax, zmax],
            ]
        )

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # bottom square
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # top square
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # vertical edges
        ]

        for i, j in edges:
            xs, ys, zs = zip(corners[i], corners[j])
            ax.plot(xs, ys, zs, "--", linewidth=2, color="g")

        # Nice aspect ratio
        ax.set_box_aspect(max_bound - min_bound)

    # Labels
    if x_title is not None:
        ax.set_xlabel(x_title)
    else:
        ax.set_xlabel("x")

    if y_title is not None:
        ax.set_ylabel(y_title)
    else:
        ax.set_ylabel("y")

    if n_dims == 3:
        ax.set_zlabel("z")

    if show:
        plt.show()

    return ax
