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
    select_greedy, select_exhaustive, swap_search, random_subset_stats,
    continuous_relays, round_to_grid, optimize_precoders, optimized_mi,
)
from relay_grid_dag.grid import (                  # scene-config defaults
    TX_XY, RX_XY, GRID_X, GRID_Y, GRID_NX, GRID_NY,
    N_TX, N_RX, N_RELAY, DIRECT_ATTEN, P_RELAY,
)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- #
# Canonical paper scenes (Sec. VII). Every single-pair experiment uses          #
# canonical_scene(); every two-pair experiment uses pair_scene(). Both are the  #
# 5x5 (L=25) candidate grid of the interactive demos.                           #
# --------------------------------------------------------------------------- #
# Two-pair terminal geometry = the 2-pair demo's INIT_POS.
PAIR_POS = [((0.0, 2.5), (20.0, 2.5)), ((0.0, -2.5), (20.0, -2.5))]
PAIR_DIRECT_ATTEN = 0.3
PAIR_CROSS_ATTEN = 0.3


def canonical_scene(model: str = "near"):
    """The frozen single-pair scene of SPEC.md Sec. 10 (= library defaults;
    the 2-pair demo shares this grid, the single-pair demo widens it): 5x5
    candidate grid on x in [6,14], y in [-4,4], Tx at (0,0) / Rx at (20,0),
    8/8/4-element ULAs, direct atten 0.3. Returns ``(scene, names, coords)``."""
    return build_candidate_scene(model)


def pair_scene(model: str = "near"):
    """The two-pair scene of the interactive 2-pair demo: same 5x5 grid,
    4/4/4-element ULAs, pairs at (0,+-2.5) -> (20,+-2.5), direct/cross atten
    0.3/0.3. Returns ``(scene, names, coords)``."""
    import relay_grid_dag.multipair as mp
    coords = mp.grid_coords(GRID_NX, GRID_NY, GRID_X, GRID_Y)
    return mp.build_pair_scene(pairs=PAIR_POS, coords=coords, model=model,
                               direct_atten=PAIR_DIRECT_ATTEN)


# --------------------------------------------------------------------------- #
# Matplotlib helpers.                                                          #
# --------------------------------------------------------------------------- #
def paper_style():
    """Shared figure style for the paper: Helvetica, slightly larger axis labels/ticks,
    and vector-PDF-friendly font embedding (call after matplotlib is imported)."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,            # embed TrueType (selectable/clean text)
        "ps.fonttype": 42,
        "figure.dpi": 120,
    })


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paper_style()
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
