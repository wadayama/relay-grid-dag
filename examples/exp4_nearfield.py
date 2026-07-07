"""EXP4 -- where the gain lives: near- vs far-field across the Rayleigh distance.

Sweep the Tx--Rx distance d with the arrays fixed (8-element ULAs, so the array
Rayleigh distance 2D^2/lambda ~ 24.5 lambda is fixed) and the 5x5 candidate grid
scaled with the link (x in [0.35d, 0.65d], y in +-0.12d). At each d we compare the
spherical-wave (near) and plane-wave (far) channel models:
  * optimized SE with K=2 greedy-selected relays (near), and the same sites under
    the far-field model (precoders re-optimized) -- the relay-grid value;
  * the direct-only SE and the direct-channel effective rank, near vs far -- the
    high-rank near-field structure that the grid exploits.

    uv run --extra examples python examples/exp4_nearfield.py [--selftest]
"""
from __future__ import annotations

import sys

import numpy as np

import _common as C
import relay_grid_dag as rgd
from relay_grid_dag import viz

K = 2
SEL_ITERS = 80
EVAL_ITERS = 250
RESTARTS = 2
DISTANCES = np.linspace(8.0, 56.0, 9)
SPACING = 0.5
RAYLEIGH = 2.0 * ((C.N_TX - 1) * SPACING) ** 2       # 24.5 lambda for 8 elements


def main(selftest=False):
    plt = C._mpl()
    dists = DISTANCES[::4] if selftest else DISTANCES
    sel_iters = 10 if selftest else SEL_ITERS
    eval_iters = 30 if selftest else EVAL_ITERS
    restarts = 1 if selftest else RESTARTS

    print(f"Rayleigh distance = {RAYLEIGH:.1f} lambda; sweeping d = "
          f"{dists[0]:.0f}..{dists[-1]:.0f}")
    opt_n, opt_f, dir_n, dir_f, rk_n, rk_f = [], [], [], [], [], []
    for d in dists:
        coords = rgd.grid_coords(5, 5, (0.35 * d, 0.65 * d), (-0.12 * d, 0.12 * d))
        sn, names, _ = rgd.build_candidate_scene("near", coords=coords,
                                                 rx_xy=(float(d), 0.0))
        sf, _, _ = rgd.build_candidate_scene("far", coords=coords,
                                             rx_xy=(float(d), 0.0))
        rk_n.append(viz.effrank(sn.channel("tx", "rx", atten=sn.direct_atten)))
        rk_f.append(viz.effrank(sf.channel("tx", "rx", atten=sf.direct_atten)))
        dir_n.append(rgd.direct_only_mi(sn))
        dir_f.append(rgd.direct_only_mi(sf))
        idx = rgd.select_greedy(sn, names, K, iters=sel_iters)
        sub = [names[i] for i in idx]
        opt_n.append(rgd.optimized_mi(sn, "tx", sub, "rx", iters=eval_iters,
                                      restarts=restarts))
        opt_f.append(rgd.optimized_mi(sf, "tx", sub, "rx", iters=eval_iters,
                                      restarts=restarts))
        print(f"  d={d:5.1f}  set={idx}  rank near/far={rk_n[-1]:.2f}/{rk_f[-1]:.2f}  "
              f"direct near/far={dir_n[-1]:5.2f}/{dir_f[-1]:5.2f}  "
              f"optimized near/far={opt_n[-1]:5.2f}/{opt_f[-1]:5.2f}")

    fig, axL = plt.subplots(figsize=(7.6, 4.5))
    axL.plot(dists, opt_n, "o-", color="tab:blue", label=f"optimized, near ($K={K}$)")
    axL.plot(dists, opt_f, "s-", color="tab:red", alpha=0.85,
             label=f"optimized, far (same sites)")
    axL.plot(dists, dir_n, "^:", color="0.35", label="direct only, near")
    axL.plot(dists, dir_f, "v:", color="0.65", label="direct only, far")
    axL.axvline(RAYLEIGH, ls="--", color="tab:purple", lw=1.3,
                label=rf"Rayleigh $\approx{RAYLEIGH:.1f}\lambda$")
    axL.set_xlabel(r"Tx--Rx distance $d_{\mathrm{TR}}$ [$\lambda$]")
    axL.set_ylabel("SE [bits/s/Hz]")
    axR = axL.twinx()
    axR.plot(dists, rk_n, "d--", color="tab:green", alpha=0.75,
             label="direct eff-rank, near")
    axR.set_ylabel("direct-channel effective rank", color="tab:green")
    axR.tick_params(axis="y", labelcolor="tab:green")
    axR.set_ylim(bottom=0.9)
    axL.legend(loc="upper right", fontsize=8.5)
    axL.grid(alpha=0.3)
    fig.tight_layout()
    p = f"{C.OUT}/exp4_nearfield.pdf"; fig.savefig(p); print("saved", p)

    ok = (np.all(np.isfinite(opt_n)) and np.all(np.array(rk_f) <= 1.0 + 1e-6)
          and np.all(np.array(opt_n) >= np.array(dir_n) - 1e-6))
    return bool(ok)


if __name__ == "__main__":
    print("exp4_nearfield: near-field value across the Rayleigh distance ...")
    raise SystemExit(0 if main(selftest="--selftest" in sys.argv) else 1)
