"""Examples-layer shared helpers (matplotlib tier).

The ported smoke scripts import this as ``C`` (a drop-in for the old
``nfprg_common``): it re-exports the library's public selection / physics API and
adds the plotting helpers and an ``out/`` directory. Keeping matplotlib here (not
in the library core) preserves the library's matplotlib-optional boundary.
"""
from __future__ import annotations

import os

import numpy as np  # noqa: F401  (re-exported convenience for scripts)
import torch        # noqa: F401

import relay_grid_dag as nfd                       # whole library, exposed as C.nfd
from relay_grid_dag import (                       # selection / grid API as C.<name>
    build_candidate_scene, grid_coords, subset_mi, direct_only_mi,
    received_power_scores, select_received_power, select_distance,
    select_greedy_mi, select_exhaustive, swap_search, random_subset_stats,
    continuous_relays, round_to_grid,
)
from relay_grid_dag.grid import (                  # scene-config defaults
    TX_XY, RX_XY, GRID_X, GRID_Y, GRID_NX, GRID_NY,
    N_TX, N_RX, N_RELAY, DIRECT_ATTEN, P_RELAY,
)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- #
# Matplotlib helpers.                                                          #
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 130})
    return plt


def plot_grid(ax, coords, *, scores=None, title=""):
    """Scatter the candidate grid (optionally coloured by ``scores``) and mark
    Tx / Rx. Returns the scatter handle (for a colorbar)."""
    sc = None
    if scores is not None:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=scores, cmap="viridis",
                        s=120, edgecolors="k", linewidths=0.4, zorder=3)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c="0.7", s=90,
                   edgecolors="k", linewidths=0.4, zorder=3, label="candidates")
    ax.scatter(*TX_XY, marker="s", s=130, c="tab:cyan", edgecolors="k",
               zorder=5, label="Tx")
    ax.scatter(*RX_XY, marker="D", s=130, c="tab:cyan", edgecolors="k",
               zorder=5, label="Rx")
    ax.set_xlabel("x [wavelengths]"); ax.set_ylabel("y [wavelengths]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, lw=0.3, alpha=0.5)
    return sc


def mark_selected(ax, coords, idx, *, color, label, marker="o", ms=22, ring=True):
    """Ring / highlight a selected index set on top of a grid plot."""
    pts = coords[list(idx)]
    if ring:
        ax.scatter(pts[:, 0], pts[:, 1], s=420, facecolors="none",
                   edgecolors=color, linewidths=2.4, marker=marker,
                   zorder=6, label=label)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], s=ms, c=color, marker=marker,
                   zorder=6, label=label)
