"""EXP0 -- system setup figure (Sec. VII-A): the two canonical scenes, to scale.

Renders the deployment geometry directly from the actual Scene objects (element
positions included), so the setup figure and every reported number derive from the
same model (consistency by construction):
  (a) single-pair canonical scene -- Tx/Rx 8-element ULAs at (0,0)/(20,0), the 5x5
      candidate grid (4-element relays) on x in [6,14] x y in [-4,4], the partially
      blocked direct link, and the array Rayleigh distance 2D^2/lambda ~ 24.5 lambda;
  (b) two-pair scene -- 4-element ULAs, pairs at (0,+-2.5) -> (20,+-2.5) sharing the
      same 5x5 grid, with the desired (solid) and cross/interference (dotted) links.

    uv run --extra examples python examples/exp0_setup.py
"""
from __future__ import annotations

import numpy as np

import _common as C

RAYLEIGH_1P = 2.0 * ((C.N_TX - 1) * 0.5) ** 2       # 8-elem, 0.5-lambda spacing
PAIR_COL = ["#d95f02", "#0072b2"]


def _elements(ax, scene, name, color, ms=2.4):
    """Plot the actual antenna-element positions of a node (from the Scene)."""
    P = scene.positions(name).detach().numpy()
    ax.plot(P[:, 0], P[:, 1], "o", ms=ms, color=color, zorder=8)


def _panel_single(ax, scene, names, coords):
    # candidate sites + their element positions
    ax.scatter(coords[:, 0], coords[:, 1], s=90, facecolors="0.88", edgecolors="k",
               linewidths=0.5, zorder=3, label=f"candidate relays (L={len(names)})")
    for nm in names:
        _elements(ax, scene, nm, "0.35", ms=1.4)
    # Tx / Rx arrays
    ax.scatter([0.0], [0.0], marker="s", s=150, c="tab:red", edgecolors="k", zorder=6,
               label="Tx (8-elem ULA)")
    ax.scatter([20.0], [0.0], marker="D", s=130, c="tab:green", edgecolors="k", zorder=6,
               label="Rx (8-elem ULA)")
    _elements(ax, scene, "tx", "tab:red")
    _elements(ax, scene, "rx", "tab:green")
    # direct link (partially blocked)
    ax.plot([0, 20], [0, 0], ls="--", lw=1.3, color="0.4", zorder=2)
    ax.annotate(r"direct link $\kappa_d=0.3$", (3.0, 0.45), ha="center", va="bottom",
                fontsize=9, color="0.3")
    # Rayleigh distance arc around the Tx
    th = np.linspace(-0.42 * np.pi, 0.42 * np.pi, 200)
    ax.plot(RAYLEIGH_1P * np.cos(th), RAYLEIGH_1P * np.sin(th), ls=":", lw=1.4,
            color="tab:purple", zorder=2,
            label=rf"Rayleigh dist. $2D_a^2/\lambda={RAYLEIGH_1P:.1f}\lambda$")
    # grid extent annotation
    from matplotlib.patches import Rectangle
    gx, gy = C.GRID_X, C.GRID_Y
    ax.add_patch(Rectangle((gx[0], gy[0]), gx[1] - gx[0], gy[1] - gy[0], fill=False,
                           ls="-", lw=0.8, edgecolor="0.6", zorder=1))
    dx = (gx[1] - gx[0]) / (C.GRID_NX - 1)
    dy = (gy[1] - gy[0]) / (C.GRID_NY - 1)
    ax.annotate(rf"$5\times5$ grid, ${dx:.0f}\lambda\times{dy:.0f}\lambda$ pitch",
                (gx[0] + (gx[1] - gx[0]) / 2, 5.35), ha="center", va="bottom",
                fontsize=9, color="0.3")
    ax.set_title("(a) single pair (canonical scene)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)


def _panel_pair(ax, scene, names, coords):
    ax.scatter(coords[:, 0], coords[:, 1], s=90, facecolors="0.88", edgecolors="k",
               linewidths=0.5, zorder=3, label=f"candidate relays (L={len(names)})")
    for nm in names:
        _elements(ax, scene, nm, "0.35", ms=1.4)
    for u, (txy, rxy) in enumerate(C.PAIR_POS):
        c = PAIR_COL[u]
        ax.scatter(*txy, marker="s", s=140, c=c, edgecolors="k", zorder=6)
        ax.scatter(*rxy, marker="D", s=120, c=c, edgecolors="k", zorder=6)
        _elements(ax, scene, f"tx{u}", c)
        _elements(ax, scene, f"rx{u}", c)
        ax.annotate(f"Tx{u}", (txy[0] - 0.5, txy[1]), ha="right", va="center",
                    fontsize=10, color=c, fontweight="bold")
        ax.annotate(f"Rx{u}", (rxy[0] + 0.5, rxy[1]), ha="left", va="center",
                    fontsize=10, color=c, fontweight="bold")
        # desired direct link
        ax.plot([txy[0], rxy[0]], [txy[1], rxy[1]], ls="-", lw=1.3, color=c,
                alpha=0.85, zorder=2)
    # cross (interference) links
    for v in range(2):
        u = 1 - v
        t, r = C.PAIR_POS[v][0], C.PAIR_POS[u][1]
        ax.plot([t[0], r[0]], [t[1], r[1]], ls=":", lw=1.3, color=PAIR_COL[v],
                alpha=0.7, zorder=2)
    ax.annotate(r"desired $\kappa_d=0.3$ (solid), cross $\kappa_x=0.3$ (dotted)",
                (10, 5.85), ha="center", va="center", fontsize=9, color="0.3")
    ax.set_title("(b) two pairs (4-elem ULAs, shared grid)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)


def main():
    plt = C._mpl()
    s1, names1, coords1 = C.canonical_scene()
    s2, names2, coords2 = C.pair_scene()
    dx = (C.GRID_X[1] - C.GRID_X[0]) / (C.GRID_NX - 1)
    dy = (C.GRID_Y[1] - C.GRID_Y[0]) / (C.GRID_NY - 1)
    print(f"single-pair: L={len(names1)}  grid pitch = {dx:.1f} x {dy:.1f} lambda  "
          f"Rayleigh(8-elem) = {RAYLEIGH_1P:.1f} lambda  (Tx-Rx = 20 lambda: near field)")
    print(f"two-pair:    L={len(names2)}  pairs at {C.PAIR_POS}")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.6, 4.4))
    _panel_single(axa, s1, names1, coords1)
    _panel_pair(axb, s2, names2, coords2)
    for ax in (axa, axb):
        ax.set_xlabel(r"x [$\lambda$]"); ax.set_ylabel(r"y [$\lambda$]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-3.0, 25.0); ax.set_ylim(-6.8, 6.8)
        ax.grid(True, lw=0.3, alpha=0.45)
    fig.tight_layout()
    p = f"{C.OUT}/exp0_setup.pdf"; fig.savefig(p); print("saved", p)
    return len(names1) == 25 and len(names2) == 25


if __name__ == "__main__":
    print("exp0_setup: system deployment figure ...")
    raise SystemExit(0 if main() else 1)
