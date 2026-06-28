"""Candidate relay grid: geometry of a Near-field Relay Grid scene.

A scene has a transmitter, a receiver, and ``L`` candidate relay sites laid out on
a grid; the selection layer (:mod:`relay_grid_dag.selection`) then activates a few
of them. The physics (channels, MI) is delegated to the vendored near-field engine.
"""
from __future__ import annotations

import numpy as np

from . import _nearfield as nfd

# --------------------------------------------------------------------------- #
# Default scene geometry (overridable per call). Lengths are in carrier         #
# wavelengths (lambda_c = 1).                                                   #
# --------------------------------------------------------------------------- #
TX_XY = (0.0, 0.0)
RX_XY = (20.0, 0.0)
GRID_X = (6.0, 14.0)
GRID_Y = (-4.0, 4.0)
GRID_NX = 5
GRID_NY = 5
N_TX = 8
N_RX = 8
N_RELAY = 4
DIRECT_ATTEN = 0.3
P_RELAY = 100.0


def grid_coords(nx: int = GRID_NX, ny: int = GRID_NY,
                xr=GRID_X, yr=GRID_Y) -> np.ndarray:
    """The ``(L, 2)`` array of candidate relay centres, row-major in ``(y, x)``."""
    xs = np.linspace(xr[0], xr[1], nx)
    ys = np.linspace(yr[0], yr[1], ny)
    return np.asarray([(float(x), float(y)) for y in ys for x in xs], dtype=float)


def build_candidate_scene(model: str = "near", *, coords: np.ndarray | None = None,
                          tx_xy=TX_XY, rx_xy=RX_XY, n_tx: int = N_TX,
                          n_rx: int = N_RX, n_relay: int = N_RELAY,
                          direct_atten: float = DIRECT_ATTEN, spacing: float = 0.5):
    """Build a scene with Tx, Rx and ``L`` candidate relay nodes ``c0..c{L-1}``.

    Returns ``(scene, names, coords)``. ``model`` is ``"near"`` or ``"far"``.
    """
    if coords is None:
        coords = grid_coords()
    s = nfd.Scene(model=model, direct_atten=direct_atten, spacing=spacing)
    s.add_node("tx", list(tx_xy), n_tx, "source")
    s.add_node("rx", list(rx_xy), n_rx, "sink")
    names = []
    for i, (x, y) in enumerate(coords):
        nm = f"c{i}"
        s.add_node(nm, [float(x), float(y)], n_relay, "relay")
        names.append(nm)
    return s, names, np.asarray(coords, dtype=float)
