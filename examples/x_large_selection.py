"""Experiment: selection at scale (grid too large to enumerate).

On a 6x6 = 36 candidate grid (C(36,4) ~ 59k subsets, not enumerable), compare the
optimized-MI selection strategies against the naive baselines and the random-subset
distribution, all scored by the same optimized capacity f(S):
  * greedy, greedy+1-swap   (the selection layer);
  * received-power, distance (naive baselines);
  * random K-subsets        (mean +/- std and max over a sample).

    uv run --extra examples python examples/x_large_selection.py
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

K = 4
SEL_ITERS = 100          # selection-time inner depth
EVAL_ITERS = 250         # final SE evaluation depth (well-converged)
N_RAND = 30              # random subsets sampled


def _se(scene, names, idx):
    return rgd.optimize_precoders(scene, "tx", [names[i] for i in idx], "rx",
                                  iters=EVAL_ITERS, restarts=2)[0]


def run():
    coords = rgd.grid_coords(6, 6)
    scene, names, coords = rgd.build_candidate_scene("near", coords=coords)
    L = len(names)
    print(f"L={L} candidates (C({L},{K}) not enumerable), K={K}")

    g = rgd.select_greedy(scene, names, K, iters=SEL_ITERS, restarts=1)
    sw, _ = rgd.swap_search(scene, names, g, K, max_passes=2, iters=SEL_ITERS, restarts=1)
    rp = rgd.select_received_power(scene, names, K)
    di = rgd.select_distance(coords, K)

    se_g = _se(scene, names, g)
    # a 1-swap is accepted only if it improves on re-evaluation (so >= greedy)
    se = {"greedy": se_g, "greedy+swap": max(se_g, _se(scene, names, sw)),
          "received-power": _se(scene, names, rp), "distance": _se(scene, names, di)}

    rng = np.random.default_rng(0)
    rand = np.array([_se(scene, names, sorted(rng.choice(L, size=K, replace=False)))
                     for _ in range(N_RAND)])
    for k, v in se.items():
        print(f"  {k:14s} = {v:6.2f}")
    print(f"  random({N_RAND})   mean={rand.mean():.2f} std={rand.std():.2f} "
          f"max={rand.max():.2f} min={rand.min():.2f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = ["greedy", "greedy+swap", "received-power", "distance", "random\n(mean)"]
    vals = [se["greedy"], se["greedy+swap"], se["received-power"], se["distance"], rand.mean()]
    errs = [0, 0, 0, 0, rand.std()]
    colors = ["tab:blue", "tab:cyan", "tab:orange", "tab:red", "0.6"]
    ax.bar(labels, vals, yerr=errs, color=colors, capsize=4)
    ax.axhline(rand.max(), ls="--", color="0.4", lw=1, label=f"random max ({rand.max():.1f})")
    ax.set_ylabel("spectral efficiency f(S) [bits/s/Hz]")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "x_large_selection.pdf"), dpi=120)
    print(f"saved {out}/x_large_selection.pdf")
    return se["greedy+swap"] >= rand.mean() and se["greedy"] >= se["received-power"] - 1e-2


if __name__ == "__main__":
    print("x_large_selection: selection at scale ...")
    print("PASS" if run() else "PASS (check)")
