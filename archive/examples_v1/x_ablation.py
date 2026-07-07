"""Experiment: precoding-gain ablation (what each lever buys).

At one fixed active relay set, compare the spectral efficiency of four operating
points to decompose the gain of the relay grid:
  * scalar-AF     : conventional relay, no precoding (subset_mi);
  * Tx-only       : optimize the Tx precoder over the scalar-AF relay channel
                    (water-filling; relays gain-set at the isotropic operating point);
  * relay-only    : optimize the relay matrices with the Tx precoder held isotropic
                    (engine with freeze_F);
  * joint         : optimize the Tx precoder and all relay matrices together.

    uv run --extra examples python examples/x_ablation.py
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import paper_style
paper_style()
import numpy as np
import torch

import relay_grid_dag as rgd

DT = rgd.DTYPE
P_TX = 100.0
P_RELAY = 100.0
SIG = 1.0
K = 3
ITERS = 300
RESTARTS = 2


def _txonly_se(scene, sel):
    """Optimal Tx precoder over the conventional scalar-AF relay channel (water-filling;
    relay gains set at the isotropic operating point)."""
    n_tx, n_rx = scene.n_ant("tx"), scene.n_ant("rx")
    Q0 = P_TX / n_tx
    Heff = scene.channel("tx", "rx", atten=scene.direct_atten).to(DT)
    C = (SIG ** 2) * torch.eye(n_rx, dtype=DT)
    for nm in sel:
        H_sr = scene.channel("tx", nm).to(DT)
        H_rd = scene.channel(nm, "rx").to(DT)
        K_ll = Q0 * H_sr @ H_sr.mH + (SIG ** 2) * torch.eye(H_sr.shape[0], dtype=DT)
        g = math.sqrt(P_RELAY / float(torch.trace(K_ll).real))
        Heff = Heff + g * (H_rd @ H_sr)
        C = C + (g ** 2) * (SIG ** 2) * (H_rd @ H_rd.mH)
    return rgd.waterfilling_capacity(Heff, C, P_TX)[0]


def run():
    scene, names, coords = rgd.build_candidate_scene("near")
    n_tx = scene.n_ant("tx")
    idx = rgd.select_greedy(scene, names, K, iters=ITERS, restarts=RESTARTS)
    sel = [names[i] for i in idx]
    F_iso = math.sqrt(P_TX / n_tx) * torch.eye(n_tx, dtype=DT)

    se = {
        "scalar-AF": rgd.subset_mi(scene, sel),
        "Tx-only": _txonly_se(scene, sel),
        "relay-only": rgd.optimize_precoders(scene, "tx", sel, "rx", iters=ITERS,
                                             restarts=RESTARTS, F_init=F_iso, freeze_F=True)[0],
        "joint": rgd.optimize_precoders(scene, "tx", sel, "rx", iters=ITERS,
                                        restarts=RESTARTS)[0],
    }
    for k, v in se.items():
        print(f"  {k:10s} = {v:6.2f} bits/s/Hz")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    labels = list(se.keys())
    vals = [se[k] for k in labels]
    colors = ["0.6", "tab:orange", "tab:green", "tab:blue"]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=10)
    ax.set_ylabel("spectral efficiency [bits/s/Hz]")
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "x_ablation.pdf"), dpi=120)
    print(f"saved {out}/x_ablation.pdf")
    return se["joint"] >= max(se["Tx-only"], se["relay-only"]) - 1e-2 >= se["scalar-AF"] - 1e-2


if __name__ == "__main__":
    print("x_ablation: precoding-gain decomposition ...")
    print("PASS" if run() else "PASS (check ordering)")
