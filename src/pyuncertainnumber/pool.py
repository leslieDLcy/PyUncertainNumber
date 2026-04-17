    # backup old plot function
    # def plot(
    #     self,
    #     title=None,
    #     ax=None,
    #     style="box",
    #     fill_color="lightgray",
    #     bound_colors=None,
    #     nuance="step",
    #     alpha=0.3,
    #     **kwargs,
    # ):
    #     """default plotting function

    #     args:
    #         style (str): 'box' or 'simple'
    #     """
    #     from .utils import CustomEdgeRectHandler

    #     if ax is None:
    #         fig, ax = plt.subplots()

    #     p_axis = self._pvalues if self._pvalues is not None else Params.p_values
    #     plot_bound_colors = bound_colors if bound_colors is not None else ["g", "b"]

    #     def display_box(nuance, label=None):
    #         """display two F curves plus the top-bottom horizontal lines"""

    #         if nuance == "step":
    #             step_kwargs = {
    #                 "c": plot_bound_colors[0],
    #                 "where": "post",
    #             }

    #             if label is not None:
    #                 step_kwargs["label"] = label

    #             # Make the plot
    #             (line,) = ax.step(self.left, p_axis, **step_kwargs)
    #             ax.step(self.right, p_axis, c=plot_bound_colors[1], where="post")
    #             ax.plot([self.left[0], self.right[0]], [0, 0], c=plot_bound_colors[1])
    #             ax.plot([self.left[-1], self.right[-1]], [1, 1], c=plot_bound_colors[0])
    #         elif nuance == "curve":
    #             smooth_curve_kwargs = {
    #                 "c": plot_bound_colors[0],
    #             }

    #             if label is not None:
    #                 smooth_curve_kwargs["label"] = label

    #             (line,) = ax.plot(self.left, p_axis, **smooth_curve_kwargs)
    #             ax.plot(self.right, p_axis, c=plot_bound_colors[1])
    #             ax.plot([self.left[0], self.right[0]], [0, 0], c=plot_bound_colors[1])
    #             ax.plot([self.left[-1], self.right[-1]], [1, 1], c=plot_bound_colors[0])
    #         else:
    #             raise ValueError("nuance must be either 'step' or 'curve'")
    #         if label is not None:
    #             ax.legend(handler_map={line: CustomEdgeRectHandler()})  # regular use

    #     if title is not None:
    #         ax.set_title(title)
    #     if style == "box":
    #         ax.fill_betweenx(
    #             y=p_axis,
    #             x1=self.left,
    #             x2=self.right,
    #             interpolate=True,
    #             color=fill_color,
    #             alpha=alpha,
    #             **kwargs,
    #         )
    #         display_box(nuance, label=None)
    #         if "label" in kwargs:
    #             ax.legend(loc="best")
    #     elif style == "simple":
    #         display_box(nuance, label=kwargs["label"] if "label" in kwargs else None)
    #     else:
    #         raise ValueError("style must be either 'simple' or 'box'")
    #     ax.set_xlabel(r"$x$")
    #     ax.set_ylabel(r"$\Pr(X \leq x)$")
    #     return ax

    # * ---- plot of p-box

    def plot(
        self,
        title=None,
        ax=None,
        style="box",
        fill_color="lightgray",
        bound_colors=None,
        bound_styles=None,  # e.g. ("--", ":")
        left_line_kwargs=None,  # e.g. {"linewidth": 2, "alpha": 0.9}
        right_line_kwargs=None,  # e.g. {"linewidth": 2, "alpha": 0.9}
        nuance="step",
        alpha=0.3,
        **kwargs,
    ):
        """default plotting function

        args:
            style (str): 'box' or 'simple'
            fill_color (str): color to fill the box (only for 'box' style)
            bound_colors (list): list of two colors for left and right bound lines
            bound_styles (list): list of two linestyles for left and right bound lines
            left_line_kwargs (dict): additional kwargs for left bound line
            right_line_kwargs (dict): additional kwargs for right bound line
            nuance (str): 'step' or 'curve' for bound line styles
            alpha (float): transparency level for the box fill (only for 'box' style)
            **kwargs: additional keyword arguments for the plot


        note:
            Two styles are supported: a 'box' with fill-in color and a 'simple' one without fill-in color.
            Color and linestyle of the bound lines can be customized via the `bound_styles`, `left_line_kwargs`, and `right_line_kwargs` parameters.
            The argument `nuance` controls whether the bound lines are plotted as step functions ('step') or smooth curves ('curve').


        example:
            >>> a = pba.normal([2, 6], [0.5, 1])
            >>> fig, ax = plt.subplots()
            >>> a.plot(ax=ax, style='simple')  # simple style without fill-in color
            >>> # box style with fill-in color and also customized bound colors
            >>> a.plot(ax=ax, style='box',
            ...     fill_color='lightblue',
            ...     bound_colors = ['lightblue', 'lightblue'],
            ...     bound_styles=("--", ":"),
            ...     alpha=0.5
            ... )
            >>> # customized left and right bound line styles
            >>> ax = pbox.plot(
            ...     left_line_kwargs={"linestyle": "--", "linewidth": 2},
            ...     right_line_kwargs={"linestyle": ":", "linewidth": 2, "alpha": 0.8},
            )

        """
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe  # optional; for "shaded/halo" line effects
        from .utils import CustomEdgeRectHandler

        if ax is None:
            fig, ax = plt.subplots()

        p_axis = self._pvalues if self._pvalues is not None else Params.p_values
        plot_bound_colors = bound_colors if bound_colors is not None else ["g", "b"]

        # defaults
        if bound_styles is None:
            bound_styles = ("solid", "solid")
        left_line_kwargs = {} if left_line_kwargs is None else dict(left_line_kwargs)
        right_line_kwargs = {} if right_line_kwargs is None else dict(right_line_kwargs)

        # ensure color + linestyle are set unless user overrode them
        left_defaults = {"c": plot_bound_colors[0], "linestyle": bound_styles[0]}
        right_defaults = {"c": plot_bound_colors[1], "linestyle": bound_styles[1]}
        # user kwargs take precedence
        left_kwargs = {**left_defaults, **left_line_kwargs}
        right_kwargs = {**right_defaults, **right_line_kwargs}

        def display_box(nuance, label=None):
            """display two F curves plus the top-bottom horizontal lines"""

            if nuance == "step":
                step_kwargs_left = {"where": "post", **left_kwargs}
                step_kwargs_right = {"where": "post", **right_kwargs}
                if label is not None:
                    step_kwargs_left["label"] = label

                (line_left,) = ax.step(self.left, p_axis, **step_kwargs_left)
                (line_right,) = ax.step(self.right, p_axis, **step_kwargs_right)
            elif nuance == "curve":
                curve_kwargs_left = {**left_kwargs}
                curve_kwargs_right = {**right_kwargs}
                if label is not None:
                    curve_kwargs_left["label"] = label

                (line_left,) = ax.plot(self.left, p_axis, **curve_kwargs_left)
                (line_right,) = ax.plot(self.right, p_axis, **curve_kwargs_right)
            else:
                raise ValueError("nuance must be either 'step' or 'curve'")

            # horizontal caps (use right/left kwargs for consistent style/color)
            ax.plot([self.left[0], self.right[0]], [0, 0], **right_kwargs)
            ax.plot([self.left[-1], self.right[-1]], [1, 1], **left_kwargs)

            if label is not None:
                ax.legend(
                    handler_map={line_left: CustomEdgeRectHandler()}
                )  # regular use

        if title is not None:
            ax.set_title(title)

        if style == "box":
            ax.fill_betweenx(
                y=p_axis,
                x1=self.left,
                x2=self.right,
                interpolate=True,
                color=fill_color,
                alpha=alpha,
                **kwargs,
            )
            display_box(nuance, label=None)
            if "label" in kwargs:
                ax.legend(loc="best")
        elif style == "simple":
            display_box(nuance, label=kwargs.get("label"))
        else:
            raise ValueError("style must be either 'simple' or 'box'")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\Pr(X \leq x)$")
        return ax
