"""Experiment: relay handover under receiver mobility (single pair).

The grid relays are fixed; as the Rx moves along a trajectory we *re-select* which K
relays to activate (greedy on the optimized MI). This is the fixed-grid, discrete
analogue of continuous relay tracking: the selection layer performs a handover.

We sweep the Rx along y (Tx fixed) and, at each frame, compare three operating modes:
  * adaptive : re-select K relays at the current Rx position (the selection layer);
  * static   : keep the frame-0 set fixed, re-optimize only its precoders;
  * direct   : the direct link alone (no relays).

Figure (3 panels): geometry (Rx trajectory + relays coloured by when they are used) |
the handover map (which relays are active per frame) | spectral efficiency vs frame.

    uv run --extra examples python examples/x_handover.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import paper_style
paper_style()
import numpy as np

import relay_grid_dag as rgd

K = 3
RX_X = 20.0
YS = np.linspace(-5.0, 5.0, 13)        # Rx trajectory along y
SEL_ITERS = 25                         # greedy-selection depth
DISP_ITERS = 250                       # precoder-optimization depth (per shown set)
RESTARTS = 2


def run():
    scene, names, coords = rgd.build_candidate_scene("near")
    L = len(names)
    idx_per, se_a, se_s, se_d = [], [], [], []
    static_set = None
    for f, y in enumerate(YS):
        scene.move("rx", [RX_X, float(y)])
        a_idx = rgd.select_greedy(scene, names, K, iters=SEL_ITERS, restarts=RESTARTS)
        if f == 0:
            static_set = a_idx
        se_a.append(rgd.optimize_precoders(scene, "tx", [names[i] for i in a_idx], "rx",
                                           iters=DISP_ITERS, restarts=RESTARTS)[0])
        se_s.append(rgd.optimize_precoders(scene, "tx", [names[i] for i in static_set], "rx",
                                           iters=DISP_ITERS, restarts=RESTARTS)[0])
        se_d.append(rgd.direct_only_mi(scene))
        idx_per.append(a_idx)
        print(f"  frame {f:2d}  Rx=({RX_X},{y:+.1f})  set={a_idx}  "
              f"SE adapt={se_a[-1]:.2f} static={se_s[-1]:.2f} direct={se_d[-1]:.2f}")

    # selection occupancy + mean-frame-of-use (for colouring)
    sel = np.zeros((len(YS), L), bool)
    for f, idx in enumerate(idx_per):
        sel[f, idx] = True
    used = sel.any(axis=0)
    mean_frame = np.array([sel[:, i].nonzero()[0].mean() if used[i] else np.nan
                           for i in range(L)])

    fig, (axg, axh, axe) = plt.subplots(1, 3, figsize=(16, 4.6))
    # --- geometry ---
    axg.scatter(coords[:, 0], coords[:, 1], s=60, c="0.8", edgecolors="k",
                linewidths=0.4, zorder=2)
    sc = axg.scatter([RX_X] * len(YS), YS, c=range(len(YS)), cmap="viridis",
                     s=60, zorder=4, label="Rx trajectory")
    ring_c = plt.cm.viridis(mean_frame[used] / max(len(YS) - 1, 1))
    axg.scatter(coords[used, 0], coords[used, 1], s=300, facecolors="none",
                edgecolors=ring_c, linewidths=2.6, zorder=5)
    axg.plot([0.0], [0.0], "r^", ms=15, zorder=6, label="Tx")
    axg.set_title("geometry: Rx path + relays used\n(ring colour = mean frame of use)")
    axg.set_xlabel(r"x [$\lambda$]"); axg.set_ylabel(r"y [$\lambda$]")
    axg.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=axg, fraction=0.046, label="frame")
    # --- handover map ---
    for f, idx in enumerate(idx_per):
        axh.scatter([f] * len(idx), idx, c=[f] * len(idx), cmap="viridis",
                    vmin=0, vmax=len(YS) - 1, s=70)
    axh.set_xlabel("frame"); axh.set_ylabel("candidate relay index")
    axh.set_title("handover map: activated relays per frame")
    axh.set_yticks(range(0, L, 4))
    # --- SE vs frame ---
    axe.plot(range(len(YS)), se_a, "o-", color="tab:blue", label="adaptive (re-select)")
    axe.plot(range(len(YS)), se_s, "s--", color="tab:orange", label="static (frame-0 set)")
    axe.plot(range(len(YS)), se_d, "^:", color="0.4", label="direct only")
    axe.set_xlabel("frame (Rx position along y)"); axe.set_ylabel("SE [bits/s/Hz]")
    axe.set_title("spectral efficiency vs Rx position")
    axe.legend(fontsize=9); axe.grid(alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    fig.suptitle("Relay handover under Rx mobility (single pair): adaptive selection vs static vs direct")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "x_handover.pdf"), dpi=120)
    gain = np.mean(np.array(se_a) - np.array(se_s))
    print(f"\nmean SE gain adaptive over static = {gain:+.2f} bits/s/Hz")
    print(f"saved {out}/x_handover.pdf")
    return bool(np.all(np.isfinite(se_a)) and np.mean(se_a) >= np.mean(se_d))


if __name__ == "__main__":
    print("x_handover: single-pair relay handover under Rx mobility ...")
    print("PASS" if run() else "FAIL")
