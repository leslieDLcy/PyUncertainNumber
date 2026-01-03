import numpy as numpy
import matplotlib as matplotlib
import matplotlib.pyplot as pyplot
from matplotlib import gridspec
from .peeling import peeling_to_structure
from .fuzzy import samples_to_fuzzy_projection
from numpy.typing import NDArray
import matplotlib.pyplot as plt

FONTSIZE = 22

FIGSIZE = {
    "small": (7, 7),
    "medium": (15, 15),
    "large": (20, 20),
}

color_dict = {
    "cyan": (0.0, 0.99, 0.99, 1),
    "yellow": (0.95, 0.95, 0, 1),
    "green": (0.0, 0.5, 0.0, 1),
    "orange": (1, 0.5, 0, 1),
    "red": (1, 0, 0, 1),
    "grey": (0.4, 0.4, 0.4, 0.8),
    "gray": (0.4, 0.4, 0.4, 0.8),
    "lightgrey": (0.7, 0.7, 0.7, 1),
    "blue": (0, 0, 0.8, 1),
    "blue2": (0, 0.4, 0.9, 1),
    "blue3": (0.5, 0.5, 0.8, 1),
    "black": (0, 0, 0, 1),
}


def c_(k, alpha=1):
    if k is None:
        t = list(color_dict["blue2"])
    elif type(k) is str:
        t = list(color_dict[k])
    else:
        if len(k) == 3:
            t = list(k) + [alpha]
        elif len(k) == 4:
            t = list(k)
    t[3] = alpha
    return tuple(t)


# cmaps = {('Perceptually Uniform Sequential',\
#             ['viridis', 'plasma', 'inferno', 'magma', 'cividis']),
#          ('Sequential',
#             ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
#          ('Sequential (2)', [
#             'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
#             'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
#             'hot', 'afmhot', 'gist_heat', 'copper']),
#          ('Diverging', [
#             'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
#             'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
#          ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
#          ('Qualitative', [
#             'Pastel1', 'Pastel2', 'Paired', 'Accent',
#             'Dark2', 'Set1', 'Set2', 'Set3',
#             'tab10', 'tab20', 'tab20b', 'tab20c']),
#          ('Miscellaneous', [
#             'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
#             'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
#             'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
#             'gist_ncar'])}

cmap = matplotlib.cm.get_cmap("tab20c")
COLORMAP = cmap.colors


def breakout(n):  # use this to determine the grid size of subplots
    def factor(n):
        t = numpy.arange(2, n, 1)
        return t[
            n % t == 0
        ]  # returns list of cofactors of an integer `n`, empty if integer `n` is prime.

    if n == 1:
        return 1, 1
    elif n == 2:
        return 1, 2
    elif n == 3:
        return 1, 3
    f = factor(n)
    while len(f) < 1:
        n += 1
        f = factor(n)
    l = len(f)
    if l == 1:
        a, b = f[0], f[0]
    elif l == 2:
        a, b = f[0], f[1]
    if l > 2:
        b = f[l // 2]
        a = n // b
    return a, b


def plot_peeling(
    x: NDArray, a, b, p=None, axes3d=False, figsize="medium", grid=True, label="X"
):
    """Plotting function for data peeling results.

    Args:
        x (NDArray): data set of iid observations
        a: sequence of subindices for each level
        b: sequence of boxes or enclosing sets
        p: upper violation probability (membership value)
    """
    x = numpy.asarray(x, dtype=float)
    n, d = x.shape
    if d == 1:
        pass
    elif d == 2:
        plot_peeling_nx2(x, a, b, p=p, figsize=figsize, grid=grid, label=label)
        return None
    elif d == 3:
        if axes3d:
            plot_peeling_3d(x, a, b, p=p, figsize=figsize, grid=grid, label=label)
        return None
    plot_peeling_nxd(
        x, a, b, p=p, figsize=figsize, grid=grid, label=label
    )  # scattermatrix with fuzzy marginals
    return None


def plot_one_fuzzy_grad(
    a_fuzzy,
    p=None,
    data=None,
    ax=None,
    figsize=None,
    grid=None,
    color=None,
    baseline_alpha=0.4,
    linewidth=0.1,
    colormap=None,
    xlabel=r"$X$",
    ylabel=r"$1-\delta$",
    flip=False,
):
    a = numpy.asarray(a_fuzzy, dtype=float)
    l = a.shape[0]
    if figsize is None:
        figsize = FIGSIZE["medium"]
    elif type(figsize) == str:
        figsize = FIGSIZE[figsize]
    fig = None
    if ax is None:
        fig, ax = pyplot.subplots(figsize=figsize)
    if p is None:
        p = numpy.arange(1 / l, 1, 1 / l)  # p = 1/n
    p = [0] + list(p)  # add ficticius zero to enable the blox plot
    alpha = numpy.linspace(baseline_alpha, 1, num=l)
    for i in range(l):
        x = [a[i, 0], a[i, 1]]
        y1 = [p[i], p[i]]  # y1=[p*i,p*i]
        y2 = [p[i + 1], p[i + 1]]  # y2=[p*(i+1),p*(i+1)]
        if colormap is not None:
            lenc = len(colormap)
            if l > lenc:
                color = colormap[i % lenc]
            else:
                color = colormap[i * (lenc // l)]
        if flip:
            ax.fill_betweenx(
                y=x,
                x1=y1,
                x2=y2,
                facecolor=c_(color, alpha=alpha[i]),
                edgecolor=c_("grey", alpha=0.8),
                linewidth=linewidth,
            )
        else:
            ax.fill_between(
                x=x,
                y1=y1,
                y2=y2,
                facecolor=c_(color, alpha=alpha[i]),
                edgecolor=c_("grey", alpha=0.8),
                linewidth=linewidth,
            )
        if data is not None:
            inside = (data >= x[0]) & (data <= x[1])
            data_level = numpy.sort(data[inside])
            if flip:
                ax.scatter(
                    2 * [p[i + 1]],
                    [data_level[0], data_level[-1]],
                    color="grey",
                    alpha=0.5,
                    s=10,
                )
            else:
                ax.scatter(
                    [data_level[0], data_level[-1]],
                    2 * [p[i + 1]],
                    color="grey",
                    alpha=0.5,
                    s=10,
                )
    if flip:
        ax.plot(
            2 * [p[i + 1]], [x[0], x[1]], color=c_("black", alpha=0.9), linewidth=0.5
        )
    else:
        ax.plot(
            [x[0], x[1]], 2 * [p[i + 1]], color=c_("black", alpha=0.9), linewidth=0.5
        )
    if data is not None:
        if flip:
            ax.scatter(len(data) * [0], data, color="grey", alpha=0.5, s=30, marker="s")
        else:
            ax.scatter(data, len(data) * [0], color="grey", alpha=0.5, s=30, marker="s")
    if grid is not None:
        ax.grid()
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    return fig, ax


def plot_box(
    box2d,
    ax=None,
    figsize=(10, 10),
    facecolor=None,
    edgecolor=None,
    alpha=None,
    label=None,
    zorder=None,
    grid=True,
):
    a, b = box2d[0], box2d[1]
    fig = None
    if ax is None:
        fig, ax = pyplot.subplots(figsize=figsize)
    ax.fill_between(
        x=[a[0], a[1]],
        y1=[b[0], b[0]],
        y2=[b[1], b[1]],
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    if grid:
        ax.grid()
    return fig, ax


def plot_peeling_nx2(
    X,
    a,
    b,
    p: list = None,
    max_level: int = None,
    label="X",
    grid=True,
    savefig: str = None,
    figsize=None,
    baseline_alpha=0.075,
):
    b = numpy.asarray(b, dtype=float)
    if figsize is None:
        figsize = FIGSIZE["medium"]
    elif type(figsize) == str:
        figsize = FIGSIZE[figsize]
    fig, axe = pyplot.subplots(
        7, 7, figsize=figsize, constrained_layout=False, sharex=True
    )
    gs = axe[0, 6].get_gridspec()
    fig.subplots_adjust(wspace=10)
    for j in range(7):
        for axi in axe[:, j]:
            axi.remove()
    axbig = fig.add_subplot(gs[2:, :5])
    ax_up = fig.add_subplot(gs[0:2, :5])
    ax_ri = fig.add_subplot(gs[2:, 5:])

    ax_up.set_ylim(0, 1)
    ax_ri.set_xlim(0, 1)

    axbig.tick_params(
        direction="out", length=6, width=0.5, labelsize=16
    )  # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html
    ax_up.tick_params(
        direction="out", length=6, width=0.5, labelsize=16
    )  # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html
    ax_ri.tick_params(
        direction="out", length=6, width=0.5, labelsize=16
    )  # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html

    axbig.scatter(X[:, 0], X[:, 1], zorder=10, marker="s")

    if max_level is None:
        L = len(a)

    alpha = baseline_alpha

    for i in range(L):
        _ = plot_box(
            b[i],
            ax=axbig,
            facecolor=c_("blue2", alpha=alpha),
            edgecolor=c_("red", alpha=alpha),
            zorder=0,
            grid=False,
        )
        axbig.scatter(X[a[i], 0], X[a[i], 1], zorder=11, marker="s", color="grey")

    _ = plot_box(
        b[L - 1],
        ax=axbig,
        facecolor=c_("blue2", alpha=alpha),
        edgecolor=c_("red", alpha=alpha),
        zorder=0,
        grid=False,
    )
    axbig.scatter(X[a[L - 1], 0], X[a[L - 1], 1], zorder=12, marker="s", color="red")

    # alpha = 0.4
    _ = plot_one_fuzzy_grad(
        b[:, 0, :], data=X[:, 0], p=p, ax=ax_up, color="blue2", baseline_alpha=alpha
    )
    _ = plot_one_fuzzy_grad(
        b[:, 1, :],
        data=X[:, 1],
        p=p,
        ax=ax_ri,
        color="blue2",
        baseline_alpha=alpha,
        flip=True,
    )

    if grid:
        axbig.grid()
        ax_up.grid()
        ax_ri.grid()

    axbig.set_aspect("auto")
    axbig.set_xlabel(r"$" + label + "_1$", fontsize=FONTSIZE)
    axbig.set_ylabel(r"$" + label + "_2$", fontsize=FONTSIZE)
    ax_up.set_xlabel(r"$" + label + "_1$", fontsize=FONTSIZE)
    ax_ri.set_ylabel(r"$" + label + "_2$", fontsize=FONTSIZE)
    ax_up.set_ylabel(r"$1-\delta$", fontsize=FONTSIZE)
    ax_ri.set_xlabel(r"$1-\delta$", fontsize=FONTSIZE)

    fig.tight_layout()
    if savefig is not None:
        fig.savefig(savefig)
    pass


def plot_peeling_3d(x, a, b):
    pass


def plot_peeling_nxd(
    x,
    a,
    b,
    fx=None,
    p: list = None,
    figsize=None,
    aspect="auto",
    label="X",
    marker="s",
    markercolor="grey",
    boxcolor="blue2",
    grid=True,
    baseline_alpha=0.075,
):
    n, d = x.shape
    l = len(b)
    if (p is None) | (fx is None):
        fx, p = peeling_to_structure(a, b, kind="scenario", beta=0.01)
    labels = [r"$" + label + "_" + str(i + 1) + "$" for i in range(d)]
    if figsize is None:
        figsize = FIGSIZE["medium"]
    elif type(figsize) == str:
        figsize = FIGSIZE[figsize]
    boxcolor = c_(boxcolor, alpha=baseline_alpha)
    markercolor = c_(markercolor, alpha=0.5)
    fig = pyplot.figure(figsize=figsize)
    gs = gridspec.GridSpec(d, d, figure=fig)
    for i in range(d):
        for j in range(d):
            ax = pyplot.subplot(gs[j, i])
            if i == j:  # on diagonal
                _ = plot_one_fuzzy_grad(
                    fx[:, i, :],
                    data=x[:, i],
                    p=p,
                    ax=ax,
                    color=boxcolor,
                    baseline_alpha=baseline_alpha,
                )
                ax.set_xlabel(labels[i], fontsize=FONTSIZE)
                ax.set_ylabel(r"$1-\delta$", fontsize=FONTSIZE)
            else:
                ax.scatter(
                    x[:, i],
                    x[:, j],
                    color=markercolor,
                    marker=marker,
                    edgecolors="face",
                )
                for box in b:
                    _ = plot_box([box[i, :], box[j, :]], ax=ax, facecolor=boxcolor)
                ax.set_aspect(aspect)
                ax.set_xlabel(labels[i], fontsize=FONTSIZE)
                ax.set_ylabel(labels[j], fontsize=FONTSIZE)
            if grid:
                ax.grid()
            ax.tick_params(
                direction="out",
                length=6,
                width=2,
                colors="#5a5a64",
                grid_color="gray",
                grid_alpha=0.5,
                labelsize="x-large",
            )
    pyplot.tight_layout()
    return ax


def plot_scattermatrix(
    x,
    bins=10,
    GS=None,
    figsize=None,
    aspect="auto",
    color=None,
    marker="s",
    alpha=None,
    edgecolors="face",
    grid=True,
    label="X",
):
    n, d = x.shape
    labels = [r"$" + label + "_" + str(i + 1) + "$" for i in range(d)]
    if figsize is None:
        figsize = FIGSIZE["medium"]
    elif type(figsize) == str:
        figsize = FIGSIZE[figsize]
    if GS is None:
        fig = pyplot.figure(figsize=figsize)
        GS = gridspec.GridSpec(d, d, figure=fig)
    SUBAX = []
    for i in range(d):
        for j in range(d):
            ax = pyplot.subplot(GS[i, j])
            if i == j:
                ax.hist(x[:, i], bins=bins, density=True, color=color)
                # ax.yaxis.tick_right()
                ax.set_xlabel(labels[i], fontsize=FONTSIZE)
                ax.set_ylabel(f"{i+1}/N", fontsize=FONTSIZE)
            else:
                ax.scatter(
                    x[:, j],
                    x[:, i],
                    color=color,
                    marker=marker,
                    alpha=alpha,
                    edgecolors=edgecolors,
                )
                ax.set_aspect(aspect)
                ax.set_xlabel(labels[j], fontsize=FONTSIZE)
                ax.set_ylabel(labels[i], fontsize=FONTSIZE)
            SUBAX.append(ax)
    for ax in SUBAX:
        if grid:
            ax.grid(True)
        ax.tick_params(
            direction="out",
            length=6,
            width=2,
            colors="#5a5a64",
            grid_color="gray",
            grid_alpha=0.5,
            labelsize="x-large",
        )
    pyplot.tight_layout()


def plot_fuzzy(
    fuzzy,
    p=None,
    data=None,
    ax=None,
    figsize=None,
    grid=False,
    color=None,
    baseline_alpha=0.4,
    linewidth=0.1,
    colormap=None,
    xlabel=None,
    ylabel=r"$1-\delta$",
    flip=False,
):
    """
    fuzzy: An (l,d_,2) array with projections
    """
    l, d_, _ = fuzzy.shape
    s1, s2 = breakout(d_)
    if xlabel is None:
        labels = [r"$X_" + str(i + 1) + "$" for i in range(d_)]
    if type(xlabel) == list:
        labels = xlabel
    if type(xlabel) == str:
        labels = [r"$" + str(xlabel) + "_" + str(i + 1) + "$" for i in range(d_)]
    if figsize is None:
        figsize = FIGSIZE["medium"]
    elif type(figsize) == str:
        figsize = FIGSIZE[figsize]
    fig, axes = pyplot.subplots(nrows=s1, ncols=s2, figsize=figsize)
    axes = numpy.matlib.repmat(axes, 1, 1)
    k = 0
    for i in range(s1):
        for j in range(s2):
            if k < d_:
                ax = axes[i, j]
                plot_one_fuzzy_grad(
                    fuzzy[:, k, :],
                    p=p,
                    ax=ax,
                    color=color,
                    baseline_alpha=baseline_alpha,
                    xlabel=labels[k],
                    ylabel=ylabel,
                )
                k += 1
                if grid:
                    ax.grid()
                ax.tick_params(
                    direction="out",
                    length=6,
                    width=2,
                    colors="#5a5a64",
                    grid_color="gray",
                    grid_alpha=0.5,
                    labelsize="x-large",
                )
    pyplot.tight_layout()
    pass


def plot_peeling_nxd_back(
    ux,
    c,
    p: list = None,
    figsize=None,
    aspect="auto",
    xlabel="X",
    ylabel=r"$1-\delta$",
    marker="s",
    markercolor="grey",
    boxcolor="blue2",
    colormap=None,
    grid=True,
    baseline_alpha=0.85,
):
    m, d_ = ux.shape
    l = len(c)
    fx = samples_to_fuzzy_projection(ux, c)
    if colormap is None:
        colormap = COLORMAP
    if p is None:
        p = numpy.arange(1 / l, 1 + 1 / l, 1 / l)  # p = 1/l
    if xlabel is None:
        labels = [r"$X_" + str(i + 1) + "$" for i in range(d_)]
    if type(xlabel) == list:
        labels = xlabel
    if type(xlabel) == str:
        labels = [r"$" + str(xlabel) + "_" + str(i + 1) + "$" for i in range(d_)]
    if figsize is None:
        figsize = FIGSIZE["medium"]
    elif type(figsize) == str:
        figsize = FIGSIZE[figsize]
    boxcolor = c_(boxcolor, alpha=baseline_alpha)
    markercolor = c_(markercolor, alpha=0.5)
    fig = pyplot.figure(figsize=figsize)
    gs = gridspec.GridSpec(d_, d_, figure=fig)
    for i in range(d_):
        for j in range(d_):
            ax = pyplot.subplot(gs[j, i])
            if i == j:  # on diagonal
                _ = plot_one_fuzzy_grad(
                    fx[:, i, :],
                    p=p,
                    ax=ax,
                    colormap=colormap,
                    baseline_alpha=baseline_alpha,
                )
                ax.set_xlabel(labels[i], fontsize=FONTSIZE)
                ax.set_ylabel(ylabel, fontsize=FONTSIZE)
            else:
                ax.scatter(ux[:, i], ux[:, j], color="gray", alpha=0.1)
                for k in range(len(c)):
                    if len(cmap.colors) < len(c):
                        colors = cmap.colors[k % len(cmap.colors)]
                    else:
                        colors = cmap.colors[k * (len(cmap.colors) // len(c))]
                    ax.scatter(
                        ux[c[k], i],
                        ux[c[k], j],
                        alpha=0.9,
                        s=20,
                        marker="s",
                        color=colors,
                    )
                ax.set_aspect(aspect)
                ax.set_xlabel(labels[i], fontsize=FONTSIZE)
                ax.set_ylabel(labels[j], fontsize=FONTSIZE)
            if grid:
                ax.grid()
            ax.tick_params(
                direction="out",
                length=6,
                width=2,
                colors="#5a5a64",
                grid_color="gray",
                grid_alpha=0.5,
                labelsize="x-large",
            )
    pyplot.tight_layout()
    pass


# * --------- refactor

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec


def _draw_peeling_cell(
    ax,
    x,
    b,
    fx,
    p,
    i,
    j,
    labels,
    aspect="auto",
    marker="s",
    markercolor="grey",
    boxcolor="blue2",
    grid=True,
    baseline_alpha=0.075,
):
    # match your styling
    boxcolor_rgba = c_(boxcolor, alpha=baseline_alpha)
    markercolor_rgba = c_(markercolor, alpha=0.5)

    if i == j:  # diagonal
        _ = plot_one_fuzzy_grad(
            fx[:, i, :],
            data=x[:, i],
            p=p,
            ax=ax,
            color=boxcolor_rgba,
            baseline_alpha=baseline_alpha,
        )
        ax.set_xlabel(labels[i], fontsize=FONTSIZE)
        ax.set_ylabel(r"$1-\delta$", fontsize=FONTSIZE)
    else:  # off-diagonal
        ax.scatter(
            x[:, i],
            x[:, j],
            color=markercolor_rgba,
            marker=marker,
            edgecolors="face",
        )
        for box in b:
            _ = plot_box([box[i, :], box[j, :]], ax=ax, facecolor=boxcolor_rgba)

        ax.set_aspect(aspect)
        ax.set_xlabel(labels[i], fontsize=FONTSIZE)
        ax.set_ylabel(labels[j], fontsize=FONTSIZE)

    if grid:
        ax.grid()

    ax.tick_params(
        direction="out",
        length=6,
        width=2,
        colors="#5a5a64",
        grid_color="gray",
        grid_alpha=0.5,
        labelsize="x-large",
    )

    return ax


def plot_peeling_nxd_all(
    x,
    a,
    b,
    fx=None,
    p: list = None,
    figsize=None,
    aspect="auto",
    label="X",
    marker="s",
    markercolor="grey",
    boxcolor="blue2",
    grid=True,
    baseline_alpha=0.075,
    return_axes=False,  # <-- new
):
    n, d = x.shape
    if (p is None) or (fx is None):
        fx, p = peeling_to_structure(a, b, kind="scenario", beta=0.01)

    labels = [r"$" + label + "_" + str(i + 1) + "$" for i in range(d)]
    if figsize is None:
        figsize = FIGSIZE["medium"]
    elif isinstance(figsize, str):
        figsize = FIGSIZE[figsize]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(d, d, figure=fig)

    axes = np.empty((d, d), dtype=object)

    for i in range(d):
        for j in range(d):
            ax = plt.subplot(gs[j, i])
            axes[j, i] = ax  # row=j, col=i (matches your gs[j,i])
            _draw_peeling_cell(
                ax=ax,
                x=x,
                b=b,
                fx=fx,
                p=p,
                i=i,
                j=j,
                labels=labels,
                aspect=aspect,
                marker=marker,
                markercolor=markercolor,
                boxcolor=boxcolor,
                grid=grid,
                baseline_alpha=baseline_alpha,
            )

    plt.tight_layout()
    return (fig, axes) if return_axes else ax  # keep backward compatible


def plot_peeling_one(
    x,
    a,
    b,
    i,
    j,
    fx=None,
    p: list = None,
    figsize=(4, 4),
    aspect="auto",
    label="X",
    marker="s",
    markercolor="grey",
    boxcolor="blue2",
    grid=True,
    baseline_alpha=0.075,
):
    n, d = x.shape
    if (p is None) or (fx is None):
        fx, p = peeling_to_structure(a, b, kind="scenario", beta=0.01)

    labels = [r"$" + label + "_" + str(k + 1) + "$" for k in range(d)]

    fig, ax = plt.subplots(figsize=figsize)
    _draw_peeling_cell(
        ax=ax,
        x=x,
        b=b,
        fx=fx,
        p=p,
        i=i,
        j=j,
        labels=labels,
        aspect=aspect,
        marker=marker,
        markercolor=markercolor,
        boxcolor=boxcolor,
        grid=grid,
        baseline_alpha=baseline_alpha,
    )
    fig.tight_layout()
    return fig, ax
