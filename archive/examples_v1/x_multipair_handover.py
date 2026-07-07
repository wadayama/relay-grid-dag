"""Experiment: relay handover under mobility with interference (2 pairs).

Pair 0 is fixed; pair 1's Rx moves along y (sweeping past pair 0's Rx, so the
interference geometry changes). On a small 3x3 grid we select K=2 relays *exhaustively*
on the optimized TIN sum-rate at each frame (adaptive = the per-frame oracle), and
compare to keeping the frame-0 set fixed (static). Because the oracle is evaluated on
the same metric, adaptive >= static by construction; the gap is the value of the
interference-aware handover as the geometry changes.

    uv run --extra examples python examples/x_multipair_handover.py
"""
from __future__ import annotations

import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import relay_grid_dag.multipair as mp
from relay_grid_dag.multipair_precoding import optimize_multipair

K = 2
ITERS = 200
RESTARTS = 1
YS = np.linspace(-5.0, 5.0, 9)        # pair-1 Rx trajectory along y
RX1_X = 20.0


def run():
    coords = mp.grid_coords(3, 3)                 # 9 candidates: exhaustive is feasible
    scene, names, coords = mp.build_pair_scene(coords=coords)
    L = len(names)
    combos = list(itertools.combinations(range(L), K))
    best_per, sum_a, sum_s = [], [], []
    static_set = None
    for f, y in enumerate(YS):
        scene.move("rx1", [RX1_X, float(y)])
        se = {c: optimize_multipair(scene, [0, 1], [names[i] for i in c],
                                    objective="sumrate", iters=ITERS, restarts=RESTARTS)[0]
              for c in combos}
        oracle = max(combos, key=lambda c: se[c])
        if f == 0:
            static_set = oracle
        sum_a.append(se[oracle])
        sum_s.append(se[static_set])
        best_per.append(list(oracle))
        print(f"  frame {f}  rx1=({RX1_X},{y:+.1f})  oracle={list(oracle)}  "
              f"sum adapt={sum_a[-1]:.2f} static={sum_s[-1]:.2f}")

    sel = np.zeros((len(YS), L), bool)
    for f, idx in enumerate(best_per):
        sel[f, idx] = True
    used = sel.any(axis=0)
    mean_frame = np.array([sel[:, i].nonzero()[0].mean() if used[i] else np.nan
                           for i in range(L)])

    fig, (axg, axe) = plt.subplots(1, 2, figsize=(12, 4.8))
    axg.scatter(coords[:, 0], coords[:, 1], s=70, c="0.8", edgecolors="k",
                linewidths=0.3, zorder=2)
    axg.plot([0.0], [2.0], "^", color="#ff7a00", ms=13, mec="k", label="Tx0")
    axg.plot([20.0], [2.0], "v", color="#ff7a00", ms=13, mec="k", label="Rx0 (fixed)")
    axg.plot([0.0], [-2.0], "^", color="#00a0d0", ms=13, mec="k", label="Tx1")
    sc = axg.scatter([RX1_X] * len(YS), YS, c=range(len(YS)), cmap="viridis", s=55,
                     zorder=4, label="Rx1 trajectory")
    ring_c = plt.cm.viridis(mean_frame[used] / max(len(YS) - 1, 1))
    axg.scatter(coords[used, 0], coords[used, 1], s=320, facecolors="none",
                edgecolors=ring_c, linewidths=2.6, zorder=5)
    axg.set_xlabel(r"x [$\lambda$]"); axg.set_ylabel(r"y [$\lambda$]")
    axg.set_title("geometry: Rx1 moves; relays used\n(ring colour = mean frame)")
    axg.legend(loc="upper left", fontsize=7); fig.colorbar(sc, ax=axg, fraction=0.046, label="frame")
    axe.plot(range(len(YS)), sum_a, "o-", color="tab:blue", label="adaptive (per-frame oracle)")
    axe.plot(range(len(YS)), sum_s, "s--", color="tab:orange", label="static (frame-0 set)")
    axe.set_xlabel("frame (Rx1 position along y)"); axe.set_ylabel("sum-rate [bits/s/Hz]")
    axe.set_title("sum-rate vs Rx1 position"); axe.legend(fontsize=9); axe.grid(alpha=0.3)
    fig.suptitle("Interference-aware relay handover (2 pairs): adaptive selection vs static")
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "x_multipair_handover.pdf"), dpi=120)
    gain = np.mean(np.array(sum_a) - np.array(sum_s))
    print(f"\nmean sum-rate gain adaptive over static = {gain:+.2f} bits/s/Hz")
    print(f"saved {out}/x_multipair_handover.pdf")
    return bool(np.all(np.array(sum_a) >= np.array(sum_s) - 1e-6))


if __name__ == "__main__":
    print("x_multipair_handover: interference-aware relay handover (exhaustive) ...")
    print("PASS" if run() else "FAIL")
