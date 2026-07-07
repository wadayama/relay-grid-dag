"""EXP6 (capstone) -- matrix relays manage interference: two pairs on the 5x5 grid.

Two Tx/Rx pairs (the 2-pair demo scene: 4-element ULAs at (0,+-2.5) -> (20,+-2.5))
share one K=2 active set, greedy-selected on the optimized TIN sum-rate at the
nominal cross gain. We sweep the cross-link (interference) attenuation and compare,
for the SAME active set:
  * matrix-optimized relays + Tx precoders (optimize_multipair), vs
  * the conventional scalar-AF relay (pair_rates),
for the TIN sum-rate and the max-min rate, with the TDMA reference
1/2 (R0_alone + R1_alone). The matrix relays' spatial DoF place interference nulls
a scalar gain cannot, so their rate barely degrades as the cross link strengthens.

    uv run --extra examples python examples/exp6_multipair.py [--selftest]
"""
from __future__ import annotations

import sys

import numpy as np

import _common as C
import relay_grid_dag.multipair as mp
from relay_grid_dag.multipair_precoding import optimize_multipair

K = 2
SEL_ITERS = 20
SUM_ITERS = 250
SUM_RESTARTS = 2
MIN_ITERS = 350           # max-min is non-smooth/harder: more iters + restarts
MIN_RESTARTS = 6
CROSS = np.linspace(0.05, 0.5, 8)
NOMINAL_CROSS = C.PAIR_CROSS_ATTEN
DIRECT = C.PAIR_DIRECT_ATTEN


def greedy_sumrate(scene, names, k, ca, iters):
    chosen, rem = [], set(range(len(names)))
    for _ in range(k):
        best, bj = -1e18, None
        for j in rem:
            act = [names[i] for i in chosen + [j]]
            v = optimize_multipair(scene, [0, 1], act, objective="sumrate",
                                   direct_atten=DIRECT, cross_atten=ca,
                                   iters=iters, restarts=1)[0]
            if v > best:
                best, bj = v, j
        chosen.append(bj); rem.discard(bj)
    return sorted(chosen)


def main(selftest=False):
    plt = C._mpl()
    scene, names, coords = C.pair_scene()
    cross = CROSS[::3] if selftest else CROSS
    sel_iters = 4 if selftest else SEL_ITERS
    sum_iters = 30 if selftest else SUM_ITERS
    sum_restarts = 1 if selftest else SUM_RESTARTS
    min_iters = 40 if selftest else MIN_ITERS
    min_restarts = 1 if selftest else MIN_RESTARTS

    idx = greedy_sumrate(scene, names, K, NOMINAL_CROSS, sel_iters)
    active = [names[i] for i in idx]
    print(f"active set (greedy sum-rate at cross={NOMINAL_CROSS}): {idx}")

    # interference-free alone rates (cross-independent) -> TDMA reference
    r0 = optimize_multipair(scene, [0], active, objective="sumrate", direct_atten=DIRECT,
                            iters=sum_iters, restarts=sum_restarts)[3][0]
    r1 = optimize_multipair(scene, [1], active, objective="sumrate", direct_atten=DIRECT,
                            iters=sum_iters, restarts=sum_restarts)[3][0]
    tdma = 0.5 * (r0 + r1)
    print(f"alone R0={r0:.2f} R1={r1:.2f}  TDMA = (R0+R1)/2 = {tdma:.2f}")

    # Max-min is non-smooth and multi-start-hungry; besides fresh random restarts we
    # warm-start each point from (i) that point's sum-rate solution and (ii) the
    # previous sweep point's max-min solution (continuation), keeping the best.
    sr_mat, sr_sca, mr_mat, mr_sca = [], [], [], []
    prev_min = None
    for ca in cross:
        kw = dict(direct_atten=DIRECT, cross_atten=float(ca))
        sr = optimize_multipair(scene, [0, 1], active, objective="sumrate",
                                iters=sum_iters, restarts=sum_restarts, **kw)
        sr_mat.append(sr[0])
        cands = [optimize_multipair(scene, [0, 1], active, objective="minrate",
                                    iters=min_iters, restarts=1,
                                    F_init=Fi, W_init=Wi, **kw)
                 for Fi, Wi in [(sr[1], sr[2])] + ([prev_min] if prev_min else [])]
        cands.append(optimize_multipair(scene, [0, 1], active, objective="minrate",
                                        iters=min_iters, restarts=min_restarts, **kw))
        mr = max(cands, key=lambda c: c[0])
        prev_min = (mr[1], mr[2])
        mr_mat.append(mr[0])
        tin = mp.pair_rates(scene, 2, active, direct_atten=DIRECT,
                            cross_atten=float(ca))["tin"]
        sr_sca.append(sum(tin)); mr_sca.append(min(tin))
        print(f"  cross={ca:.2f}  sum: matrix={sr_mat[-1]:6.2f} scalar-AF={sr_sca[-1]:6.2f} | "
              f"min: matrix={mr_mat[-1]:5.2f} (R={[round(t,2) for t in mr[3]]}) "
              f"scalar-AF={mr_sca[-1]:5.2f}")

    fig, (axs, axm) = plt.subplots(1, 2, figsize=(11.8, 4.3))
    axs.plot(cross, sr_mat, "o-", color="tab:blue", label="matrix relays (optimized)")
    axs.plot(cross, sr_sca, "s--", color="tab:orange", label="conventional scalar-AF")
    axs.axhline(tdma, ls=":", color="0.4",
                label=rf"TDMA $=\frac{{1}}{{2}}(R_0+R_1)_{{\rm alone}}={tdma:.1f}$")
    axs.set_xlabel(r"cross-link gain $\kappa_x$ (interference)")
    axs.set_ylabel("TIN sum-rate [bits/s/Hz]")
    axs.set_title("(a) sum-rate vs interference")
    axs.legend(fontsize=9); axs.grid(alpha=0.3); axs.set_ylim(bottom=0)
    axm.plot(cross, mr_mat, "o-", color="tab:blue", label="matrix relays (optimized)")
    axm.plot(cross, mr_sca, "s--", color="tab:orange", label="conventional scalar-AF")
    axm.set_xlabel(r"cross-link gain $\kappa_x$ (interference)")
    axm.set_ylabel("min-rate [bits/s/Hz]")
    axm.set_title("(b) max-min rate vs interference")
    axm.legend(fontsize=9); axm.grid(alpha=0.3); axm.set_ylim(bottom=0)
    fig.tight_layout()
    p = f"{C.OUT}/exp6_multipair.pdf"; fig.savefig(p); print("saved", p)

    return bool(np.all(np.array(sr_mat) >= np.array(sr_sca) - 1e-2)
                and np.all(np.isfinite(mr_mat)))


if __name__ == "__main__":
    print("exp6_multipair: matrix vs scalar-AF relays under interference ...")
    raise SystemExit(0 if main(selftest="--selftest" in sys.argv) else 1)
