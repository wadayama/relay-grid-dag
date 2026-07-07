"""EXP7 (capstone figure) -- power is not rate: field vs per-stream MI maps.

Reuses the 2-pair interactive demo's model verbatim (same 5x5 grid, same terminal
positions), so the published figure IS the demo's model at publication optimization
depth. From ONE set of optimized precoders (K=2, TIN sum-rate) it renders:
  * the total carrier field [dB] -- where the transmitted *power* goes;
  * the EXACT per-pixel array MI maps I(X0;Y), I(X1;Y) [bits] via the K-recursion --
    the rate a receive array at each point would get, treating the other stream as
    noise.
The MI maps focus on each pair's own Rx and collapse at the other's (the
interference null) -- structure invisible in the power map.

    uv run --extra examples python examples/exp7_mimap.py [--selftest]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

import _common as C

sys.path.insert(0, os.path.dirname(__file__))
_ARGS = sys.argv; sys.argv = ["x"]          # keep the demo module import inert
import relay_grid_demo_2pair as D           # noqa: E402
sys.argv = _ARGS

import relay_grid_dag as rgd                # noqa: E402

SEL_ITERS = 20
DISP_ITERS = 250
DISP_RESTARTS = 3


def _set_resolution(nx, ny):
    D.HEAT_NX, D.HEAT_NY = nx, ny
    gx, gy = np.meshgrid(np.linspace(*D.PLANE_X, nx), np.linspace(*D.PLANE_Y, ny))
    D._GRID = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=rgd.RDTYPE)
    D._ZERO = np.zeros((ny, nx))


def main(selftest=False):
    plt = C._mpl()
    D.SEL_ITERS = 4 if selftest else SEL_ITERS
    D.DISP_ITERS = 30 if selftest else DISP_ITERS
    D.DISP_RESTARTS = 1 if selftest else DISP_RESTARTS
    if selftest:
        _set_resolution(36, 26)

    pos = dict(D.INIT_POS)
    s, names = D.build_scene(pos)
    idx = D.select_set(s, names, 2, [True, True], "sumrate", D.P_RELAY)
    rates, intens, F, W, active = D.render(s, names, idx, [True, True], "sumrate",
                                           D.P_RELAY)
    print(f"set={idx}  R0={rates[0]:.2f}  R1={rates[1]:.2f}  "
          f"sum={sum(rates.values()):.2f} bits/s/Hz")

    panels = [D.display_image("field", intens),
              D.exact_mi_map(s, active, F, W, [True, True], 0, pos["rx0"]),
              D.exact_mi_map(s, active, F, W, [True, True], 1, pos["rx1"])]
    titles = ["(a) total carrier field [dB]",
              r"(b) MI map $I(X_0;Y)$ [bits]", r"(c) MI map $I(X_1;Y)$ [bits]"]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.3))
    for ax, (arr, vmn, vmx, _), ti in zip(axes, panels, titles):
        im = ax.imshow(arr, origin="lower", extent=[*D.PLANE_X, *D.PLANE_Y],
                       aspect="auto", cmap=D.CMAP, vmin=vmn, vmax=vmx, zorder=0,
                       interpolation="bilinear")
        D._draw(ax, pos, idx, [True, True], ms=10)
        for u in range(D.M):
            for i in idx:
                ax.plot([pos[f"tx{u}"][0], D._COORDS[i, 0], pos[f"rx{u}"][0]],
                        [pos[f"tx{u}"][1], D._COORDS[i, 1], pos[f"rx{u}"][1]],
                        color=D.PAIR_COL[u], ls="--", lw=1.0, alpha=0.65, zorder=5)
        ax.set_xlim(*D.PLANE_X); ax.set_ylim(*D.PLANE_Y)
        ax.set_xlabel(r"x [$\lambda$]"); ax.set_ylabel(r"y [$\lambda$]")
        ax.set_title(ti, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = f"{C.OUT}/exp7_mimap.pdf"; fig.savefig(p); print("saved", p)

    mi0 = panels[1][0]; mi1 = panels[2][0]
    return bool(all(v > 0 for v in rates.values())
                and np.isfinite(mi0).all() and np.isfinite(mi1).all())


if __name__ == "__main__":
    print("exp7_mimap: carrier field vs exact MI maps (2 pairs) ...")
    raise SystemExit(0 if main(selftest="--selftest" in _ARGS) else 1)
