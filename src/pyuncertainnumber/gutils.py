# * ---------------------helper functions  --------------------- *#
from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .pba.pbox_abc import Pbox
    from .pba.dss import DempsterShafer


def inspect_un(x: any):
    """Inspect the any type of uncertain number x."""
    print(x.__repr__())
    x.display()


def env_helper(elements: list, env):
    """help visualise the envelope"""

    fig, ax = plt.subplots()

    env.plot(fill_color="salmon", ax=ax, bound_colors=["salmon", "salmon"], zorder=50)

    for p in elements:
        p.plot(ax=ax, zorder=100)
