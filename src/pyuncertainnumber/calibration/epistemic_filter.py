from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from functools import partial


class EpistemicFilter:
    """The EpistemicFilter method to reduce the epistemic uncertainty space based on discrepancy scores.

    args:
        xe_samples (NDArray): Proposed Samples of epistemic parameters, shape (ne, n_dimensions).
            Typically samples from a bounded set of some epistemic parameters.

        discrepancy_scores (NDArray, optional): Discrepancy scores between the model simulations and the observation.
            Associated with each xe sample, shape (ne,). Defaults to None.

        sets_of_discrepancy (list, optional): List of sets of discrepancy scores for multiple datasets.
            Each element should be an NDArray of shape (ne,). Defaults to None.

    tip:
        For performance functions that output multiple responses, some aggregation of discrepancy scores may be used.
        Depending on the number of observation, either a single set of discrepancy scores or multiple sets can be provided.

    .. figure:: /_static/convex_hull.png
        :alt: convex hull with bounds
        :align: center
        :width: 50%

        Convex hull with bounds illustration.
    """

    def __init__(
        self,
        xe_samples: NDArray,
        discrepancy_scores: NDArray = None,
        sets_of_discrepancy: list = None,
    ):

        self.xe_samples = xe_samples
        self.discrepancy_scores = discrepancy_scores
        self.sets_of_discrepancy = (
            sets_of_discrepancy if sets_of_discrepancy is not None else None
        )

    def filter(self, threshold: float | list):
        """Filter the epistemic samples based on a discrepancy threshold.

        args:
            threshold (float | list): The discrepancy threshold(s) for filtering data points.

        returns:
            tuple:
                - the filtered xe samples;
                - hull;
                - lower bounds;
                - upper bounds of the bounding box, or (None, None) if unsuccessful.
        """
        if isinstance(threshold, list):
            return self.iterative_filter(thresholds=threshold)
        else:
            return filter_by_discrepancy(
                self.xe_samples, self.discrepancy_scores, threshold
            )

    def iterative_filter(
        self,
        thresholds: list,
    ) -> list:
        """Iteratively filter the epistemic samples based on multiple thresholds.

        args:
            thresholds (list): List of discrepancy thresholds for filtering data points.

            simulation_model (callable, optional): A function that takes xe_samples as input and returns discrepancy scores.
                Required if re_sample is True. Defaults to None.

        note:
            thresholds are assumed to be sorted in ascending order.
        """

        f = partial(
            filter_by_discrepancy,
            xe_samples=self.xe_samples,
            discrepancy_scores=self.discrepancy_scores,
        )
        return [f(t) for t in thresholds.sort()]

    # TODO develope this function which natively intergrates the simulation model and resampling
    def iterative_filter_w_simulation_model(
        self, thresholds: list, re_sample: bool = False, simulation_model=None
    ):
        """Iteratively filter the epistemic samples based on multiple thresholds.

        args:
            thresholds (list): List of discrepancy thresholds for filtering data points.

            re_sample (bool, optional): If True, re-evaluate discrepancy scores at each iteration using the simulation_model.
                Defaults to False.

            simulation_model (callable, optional): A function that takes xe_samples as input and returns discrepancy scores.
                Required if re_sample is True. Defaults to None.
        note:
            if re_sample is True, the simulation_model should be provided to re-evaluate discrepancy scores.
        """

        pass

    def filter_on_sets(self, plot_hulls: bool = True):
        """Filter the epistemic samples based on multiple sets of discrepancy scores.

        args:
            plot_hulls (bool, optional): If True, plots the convex hulls of the first five sets. Defaults to True.

        returns:
            tuple:
                - lower bounds;
                - upper bounds of the intersected bounding box, or (None, None) if unsuccessful

        note:
            `self.sets_of_discrepancy` must exist.
        """

        xe_list = []
        hull_list = []
        boxes = []

        for i in range(len(self.sets_of_discrepancy)):
            ef = EpistemicFilter(
                xe_samples=self.xe_samples,
                discrepancy_scores=self.sets_of_discrepancy[i],
            )
            filtered_xe, hull, low_bound, upper_bound = ef.filter(threshold=10)
            xe_list.append(filtered_xe)
            hull_list.append(hull)
            boxes.append((hull.min_bound, hull.max_bound))

        if plot_hulls:
            plot_multiple_convex_hulls(xe_list[:5], hull_list[:5])

        # Compute intersection of bounding boxes
        box_int_lower = np.maximum(boxes[0][0], boxes[1][0])
        box_int_upper = np.minimum(boxes[0][1], boxes[1][1])

        return box_int_lower, box_int_upper

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


# problem agnostic
def filter_by_discrepancy(xe_samples, discrepancy_scores, threshold=0.1) -> tuple:
    """Computes the intersection of convex hull bounding boxes based on a discrepancy threshold.

    args:
        threshold (float): The scalar discrepancy threshold for filtering data points.

    returns:
        tuple: the hull, and Lower and upper bounds of the intersected bounding box, or (None, None) if unsuccessful.

    note:
        Assume the threshold is a scalar value.
    """
    # Get the absolute path of the directory containing this script

    filtered_xe = xe_samples[discrepancy_scores < threshold]

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
    filtered_xe: NDArray,
    hull=None,
    ax=None,
    show=True,
    x_title=None,
    y_title=None,
    hull_alpha=0.25,
):
    """Plot points, their convex hull, and the axis-aligned bounding box.

    args:
    filtered_xe (NDArray): array-like, shape (n_samples, n_dims)
        Input points. Supports 2D or 3D.

    hull : scipy.spatial.ConvexHull, optional
        Precomputed convex hull. If None, it is computed.

    ax : matplotlib Axes or 3D Axes, optional
        Existing axes to draw on. If None, a new figure/axes is created.

    show : bool, default True
        If True, calls plt.show() at the end.

    hull_alpha : float, default 0.25
        Transparency of the hull surface.

    Returns:
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


def intersect_convex_hulls(hulls):
    """Compute intersection of several ConvexHull polytopes.
    Works in 2D or 3D (and in principle higher).

    args:
        hulls : list of scipy.spatial.ConvexHull

    returns:
        vertices : (m, d) ndarray
            Vertices of the intersection polytope (may be empty).
        hull_int : ConvexHull or None
            ConvexHull of the intersection vertices, or None if empty.
    """
    if not hulls:
        raise ValueError("Need at least one hull.")

    dim = hulls[0].points.shape[1]

    # 1. Collect all halfspaces: a^T x + c <= 0
    halfspaces = np.vstack([h.equations for h in hulls])

    # 2. Try to find a feasible interior point by linear programming
    A = halfspaces[:, :-1]
    c = halfspaces[:, -1]

    # HalfspaceIntersection expects a^T x + c <= 0.
    # We solve A x + c <= 0  <=>  A x <= -c
    res = linprog(
        np.zeros(dim),
        A_ub=A,
        b_ub=-c,
        method="highs",
    )

    if not res.success:
        # Intersection is empty (or numerical issues)
        return np.empty((0, dim)), None

    interior_point = res.x

    # 3. Build the halfspace intersection
    hs_int = HalfspaceIntersection(halfspaces, interior_point)
    vertices = hs_int.intersections

    if len(vertices) == 0:
        return vertices, None

    hull_int = ConvexHull(vertices)
    return vertices, hull_int


def plot_multiple_convex_hulls(
    xe_list,
    hulls=None,
    show_bounds=True,
    hull_alpha=0.25,
    ax=None,
    show=True,
):
    """Plot multiple 2D point sets with their convex hulls.

    args:
        xe_list : list of ndarray
            Each array must be shape (n_i, 2)
        hulls : list of ConvexHull or None
            If None, hulls are computed automatically.
        show_bounds : bool
            Whether to draw dashed bounding rectangles.
        hull_alpha : float
            Transparency of hull fill.
        ax : matplotlib Axes
            Optional existing axes.
        show : bool
            Whether to call plt.show()

    Returns:
        fig, ax, hulls
    """

    # Convert inputs and validate dimensions
    xe_list = [np.asarray(xe) for xe in xe_list]
    for xe in xe_list:
        if xe.ndim != 2 or xe.shape[1] != 2:
            raise ValueError("All point sets must be 2D with shape (n_i, 2).")

    # Determine hulls if needed
    if hulls is None:
        hulls = []
        for xe in xe_list:
            if xe.shape[0] < 3:  # not enough points for a hull
                hulls.append(None)
            else:
                hulls.append(ConvexHull(xe))

    # Prepare figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Plot each convex hull
    for idx, (xe, hull) in enumerate(zip(xe_list, hulls)):
        color = f"C{idx % 10}"

        # Plot points
        ax.scatter(xe[:, 0], xe[:, 1], s=20, color=color)

        # Draw the hull polygon
        if hull is not None:
            verts = xe[hull.vertices]
            verts_closed = np.vstack([verts, verts[0]])

            ax.fill(
                verts_closed[:, 0],
                verts_closed[:, 1],
                facecolor=color,
                edgecolor=color,
                alpha=hull_alpha,
                linewidth=2,
            )

            # Bounding rectangle
            if show_bounds:
                min_bound = getattr(hull, "min_bound", xe.min(axis=0))
                max_bound = getattr(hull, "max_bound", xe.max(axis=0))
                xmin, ymin = min_bound
                xmax, ymax = max_bound

                rect_x = [xmin, xmax, xmax, xmin, xmin]
                rect_y = [ymin, ymin, ymax, ymax, ymin]
                ax.plot(rect_x, rect_y, "--", linewidth=1.5, color=color)

    # ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if show:
        plt.show()

    return ax
