"""Experiment: near-field value vs link distance (Rayleigh crossing).

Sweep the Tx--Rx distance (arrays fixed, so the array Rayleigh distance is fixed at
~2D^2/lambda). The relay grid is scaled to stay between Tx and Rx. We plot, vs
distance: the optimized SE (K=2 relays, joint precoding), the direct-only SE, and the
effective rank of the direct channel. Inside the Rayleigh distance the spherical
wavefront keeps the channel high-rank (big SE / relay gain); beyond it the channel
collapses toward the rank-one far-field limit.

    uv run --extra examples python examples/x_distance_sweep.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import paper_style
paper_style()
import numpy as np
import torch

import relay_grid_dag as rgd

K = 2
SEL_ITERS = 120
DISP_ITERS = 250
SPACING = 0.5
N_TX = 8
DISTANCES = np.linspace(8.0, 56.0, 9)
RAYLEIGH = 2.0 * ((N_TX - 1) * SPACING) ** 2          # 2 D^2 / lambda (lambda=1)


def _effrank(H):
    s = torch.linalg.svdvals(H).real
    return float((s.sum() ** 2) / (s ** 2).sum())


def run():
    print(f"Rayleigh distance ~ {RAYLEIGH:.1f} (lambda); sweeping Tx-Rx distance")
    se_opt, se_dir, rank = [], [], []
    for d in DISTANCES:
        coords = rgd.grid_coords(3, 3, (0.35 * d, 0.65 * d), (-0.12 * d, 0.12 * d))
        scene, names, _ = rgd.build_candidate_scene("near", coords=coords,
                                                    rx_xy=(float(d), 0.0), n_tx=N_TX,
                                                    n_rx=N_TX, spacing=SPACING)
        H = scene.channel("tx", "rx", atten=scene.direct_atten)
        rank.append(_effrank(H))
        se_dir.append(rgd.direct_only_mi(scene))
        idx = rgd.select_greedy(scene, names, K, iters=SEL_ITERS, restarts=1)
        se_opt.append(rgd.optimize_precoders(scene, "tx", [names[i] for i in idx], "rx",
                                             iters=DISP_ITERS, restarts=2)[0])
        print(f"  d={d:5.1f}  effrank={rank[-1]:.2f}  direct={se_dir[-1]:5.2f}  "
              f"optimized={se_opt[-1]:5.2f}  gain={se_opt[-1]-se_dir[-1]:5.2f}")

    fig, axL = plt.subplots(figsize=(7.4, 4.6))
    axL.plot(DISTANCES, se_opt, "o-", color="tab:blue", label="optimized SE (K=2)")
    axL.plot(DISTANCES, se_dir, "^:", color="0.4", label="direct only")
    axL.axvline(RAYLEIGH, ls="--", color="tab:red", lw=1.2,
                label=f"Rayleigh ~{RAYLEIGH:.0f}")
    axL.set_xlabel(r"Tx--Rx distance [$\lambda$]")
    axL.set_ylabel("spectral efficiency [bits/s/Hz]")
    axR = axL.twinx()
    axR.plot(DISTANCES, rank, "s--", color="tab:green", alpha=0.7, label="direct eff-rank")
    axR.set_ylabel("direct-channel effective rank", color="tab:green")
    axR.tick_params(axis="y", labelcolor="tab:green")
    axL.legend(loc="upper right", fontsize=8)
    axL.grid(alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "x_distance_sweep.pdf"), dpi=120)
    print(f"saved {out}/x_distance_sweep.pdf")
    return bool(np.all(np.isfinite(se_opt)))


if __name__ == "__main__":
    print("x_distance_sweep: near-field value vs distance ...")
    print("PASS" if run() else "FAIL")
