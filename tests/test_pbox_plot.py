import matplotlib.pyplot as plt
import pytest

from pyuncertainnumber import pba


@pytest.fixture(scope="module")
def imprecise_beta_pbox():
    return pba.beta([2.498, 4.015], [6.267, 9.158])


def test_plot_alpha_applies_to_fill_and_bounds(imprecise_beta_pbox):
    fig, ax = plt.subplots()
    try:
        imprecise_beta_pbox.plot(
            ax=ax,
            fill_color="coral",
            bound_colors=["snow", "snow"],
            label="TMCMC",
            nuance="curve",
            alpha=0.2,
        )

        assert ax.collections[0].get_alpha() == pytest.approx(0.2)
        assert len(ax.lines) == 4
        assert all(line.get_alpha() == pytest.approx(0.2) for line in ax.lines)
    finally:
        plt.close(fig)


def test_plot_bound_alpha_can_override_shared_alpha(imprecise_beta_pbox):
    fig, ax = plt.subplots()
    try:
        imprecise_beta_pbox.plot(
            ax=ax,
            nuance="curve",
            alpha=0.2,
            left_line_kwargs={"alpha": 0.6},
            right_line_kwargs={"alpha": 0.8},
        )

        assert [line.get_alpha() for line in ax.lines] == pytest.approx(
            [0.6, 0.8, 0.8, 0.6]
        )
    finally:
        plt.close(fig)
