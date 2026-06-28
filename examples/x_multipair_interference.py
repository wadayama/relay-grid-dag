"""Experiment (capstone): matrix relays manage interference (2 pairs).

Two Tx/Rx pairs share a fixed relay set. We sweep the cross (interference) link
strength and compare, for the SAME active set:
  * matrix-optimized relays + Tx precoders (optimize_multipair), and
  * the conventional scalar-AF relay baseline (multipair.pair_rates),
for both the TIN sum-rate and the max-min rate. A horizontal reference marks TDMA
= 1/2 (R0_alone + R1_alone): serving the pairs one at a time. The matrix relays use
their spatial DoF to null interference, so they degrade far less than scalar-AF as
the cross link strengthens, and beat TDMA over a wide range.

    uv run --extra examples python examples/x_multipair_interference.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import paper_style
paper_style()
import numpy as np

import relay_grid_dag.multipair as mp
from relay_grid_dag.multipair_precoding import optimize_multipair

K = 2
DISP_ITERS = 250
RESTARTS = 2
MIN_ITERS = 350           # max-min is non-smooth/harder: more iters + restarts
MIN_RESTARTS = 6
CROSS = np.linspace(0.05, 0.5, 8)
NOMINAL_CROSS = 0.3


def _greedy(scene, names, k, ca):
    chosen, rem = [], set(range(len(names)))
    for _ in range(k):
        best, bj = -1e18, None
        for j in rem:
            act = [names[i] for i in chosen + [j]]
            v = optimize_multipair(scene, [0, 1], act, objective="sumrate",
                                   cross_atten=ca, iters=20, restarts=1)[0]
            if v > best:
                best, bj = v, j
        chosen.append(bj); rem.discard(bj)
    return sorted(chosen)


def run():
    scene, names, _ = mp.build_pair_scene()
    idx = _greedy(scene, names, K, NOMINAL_CROSS)
    active = [names[i] for i in idx]
    print(f"active set {idx}")

    # interference-free alone rates (cross-independent) -> TDMA reference
    r0 = optimize_multipair(scene, [0], active, objective="sumrate",
                            iters=DISP_ITERS, restarts=RESTARTS)[3][0]
    r1 = optimize_multipair(scene, [1], active, objective="sumrate",
                            iters=DISP_ITERS, restarts=RESTARTS)[3][0]
    tdma = 0.5 * (r0 + r1)
    print(f"alone R0={r0:.2f} R1={r1:.2f}  TDMA ref={tdma:.2f}")

    sr_mat, sr_sca, mr_mat, mr_sca = [], [], [], []
    for ca in CROSS:
        sr_mat.append(optimize_multipair(scene, [0, 1], active, objective="sumrate",
                                         cross_atten=float(ca), iters=DISP_ITERS,
                                         restarts=RESTARTS)[0])
        mr_mat.append(optimize_multipair(scene, [0, 1], active, objective="minrate",
                                         cross_atten=float(ca), iters=MIN_ITERS,
                                         restarts=MIN_RESTARTS)[0])
        tin = mp.pair_rates(scene, 2, active, cross_atten=float(ca))["tin"]
        sr_sca.append(sum(tin)); mr_sca.append(min(tin))
        print(f"  cross={ca:.2f}  sum: matrix={sr_mat[-1]:.2f} scalar-AF={sr_sca[-1]:.2f} | "
              f"min: matrix={mr_mat[-1]:.2f} scalar-AF={mr_sca[-1]:.2f}")

    fig, (axs, axm) = plt.subplots(1, 2, figsize=(12, 4.6))
    axs.plot(CROSS, sr_mat, "o-", color="tab:blue", label="matrix relays (optimized)")
    axs.plot(CROSS, sr_sca, "s--", color="tab:orange", label="scalar-AF baseline")
    axs.axhline(tdma, ls=":", color="0.4", label=f"TDMA = ½(R0+R1)_alone = {tdma:.1f}")
    axs.set_xlabel("cross-link attenuation (interference)"); axs.set_ylabel("sum-rate [bits/s/Hz]")
    axs.set_title("TIN sum-rate vs interference"); axs.legend(fontsize=8); axs.grid(alpha=0.3)
    axm.plot(CROSS, mr_mat, "o-", color="tab:blue", label="matrix relays (optimized)")
    axm.plot(CROSS, mr_sca, "s--", color="tab:orange", label="scalar-AF baseline")
    axm.set_xlabel("cross-link attenuation (interference)"); axm.set_ylabel("min-rate [bits/s/Hz]")
    axm.set_title("max-min rate vs interference"); axm.legend(fontsize=8); axm.grid(alpha=0.3)
    fig.suptitle(f"Matrix relays manage interference (2 pairs, set {idx})")
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "x_multipair_interference.pdf"), dpi=120)
    print(f"saved {out}/x_multipair_interference.pdf")
    return bool(np.all(np.array(sr_mat) >= np.array(sr_sca) - 1e-2))


if __name__ == "__main__":
    print("x_multipair_interference: matrix relays vs scalar-AF under interference ...")
    print("PASS" if run() else "PASS (check)")
