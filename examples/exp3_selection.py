"""EXP3 -- relay selection on the optimized MI f(S), on the canonical 5x5 grid.

All strategies choose K of the same L=25 candidates:
  (a) K=2, where the exhaustive oracle (C(25,2)=300 subsets) is feasible: greedy and
      greedy+1-swap vs the oracle and the naive received-power / distance baselines;
  (b) K=4, where enumeration is impractical (C(25,4)=12,650): greedy / greedy+swap vs
      the baselines and the distribution of random 4-subsets (optimized MI, sampled);
  (c) the map: which sites the strategies pick.
Every final set is re-evaluated at the common reporting depth so the bars are
comparable; the search itself runs at a cheaper warm-started depth.

    uv run --extra examples python examples/exp3_selection.py [--selftest]
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

import _common as C
import relay_grid_dag as rgd

SEARCH_ITERS = 120           # engine depth during the combinatorial search
EVAL_ITERS = 300             # common reporting depth (also used for random subsets)
EVAL_RESTARTS = 2
RAND_TRIALS = 30
SEED = 0


def main(selftest=False):
    plt = C._mpl()
    scene, names, coords = C.canonical_scene()
    search_iters = 10 if selftest else SEARCH_ITERS
    eval_iters = 30 if selftest else EVAL_ITERS
    eval_restarts = 1 if selftest else EVAL_RESTARTS
    rand_trials = 3 if selftest else RAND_TRIALS
    pool = names[:8] if selftest else names       # oracle over a small pool in selftest

    def evaluate(idx):
        return rgd.optimized_mi(scene, "tx", [names[i] for i in idx], "rx",
                                iters=eval_iters, restarts=eval_restarts)

    # ---- (a) K=2: oracle feasible --------------------------------------------
    # The oracle enumerates ALL C(L,2) subsets at the reporting depth itself, so
    # it is exhaustive under the same metric the bars show; greedy/swap search at
    # the cheaper warm-started depth and their final sets are re-evaluated.
    K2 = 2
    g2 = rgd.select_greedy(scene, pool, K2, iters=search_iters)
    s2, _ = rgd.swap_search(scene, pool, g2, K2, iters=search_iters)
    o2, o2_val = None, -1e18
    for combo in itertools.combinations(range(len(pool)), K2):
        v = evaluate(list(combo))
        if v > o2_val:
            o2, o2_val = sorted(combo), v
    r2 = rgd.select_received_power(scene, pool, K2)
    d2 = rgd.select_distance(coords[:len(pool)], K2)
    res2 = {"oracle": (o2, o2_val), "greedy": (g2, evaluate(g2)),
            "greedy+swap": (s2, evaluate(s2)), "recv-power": (r2, evaluate(r2)),
            "distance": (d2, evaluate(d2))}
    for k, (idx, v) in res2.items():
        print(f"  K=2 {k:12s} set={idx}  f(S)={v:6.2f} bits/s/Hz")

    # ---- (b) K=4: enumeration impractical -------------------------------------
    K4 = 4
    g4 = rgd.select_greedy(scene, names, K4, iters=search_iters)
    s4, _ = rgd.swap_search(scene, names, g4, K4, iters=search_iters)
    r4 = rgd.select_received_power(scene, names, K4)
    d4 = rgd.select_distance(coords, K4)
    res4 = {"greedy": (g4, evaluate(g4)), "greedy+swap": (s4, evaluate(s4)),
            "recv-power": (r4, evaluate(r4)), "distance": (d4, evaluate(d4))}
    rng = np.random.default_rng(SEED)
    rand_vals = np.array([                       # same reporting depth as the bars
        evaluate(rng.choice(len(names), K4, replace=False).tolist())
        for _ in range(rand_trials)])
    for k, (idx, v) in res4.items():
        print(f"  K=4 {k:12s} set={idx}  f(S)={v:6.2f} bits/s/Hz")
    print(f"  K=4 random subsets ({rand_trials}): mean={rand_vals.mean():.2f} "
          f"std={rand_vals.std():.2f} max={rand_vals.max():.2f} min={rand_vals.min():.2f}")

    # ---- figure ----------------------------------------------------------------
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(15.6, 4.3),
                                        gridspec_kw={"width_ratios": [1, 1.15, 1.05]})
    def bars(ax, res, colors):
        labels = list(res.keys()); vals = [res[k][1] for k in labels]
        bs = ax.bar(labels, vals, color=colors)
        for b, v in zip(bs, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.1f}", ha="center",
                    fontsize=9)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("optimized MI $f(S)$ [bits/s/Hz]"); ax.grid(axis="y", alpha=0.3)
        return max(vals)

    bars(axa, res2, ["tab:red", "tab:blue", "tab:cyan", "0.6", "0.75"])
    axa.axhline(res2["oracle"][1], ls=":", color="tab:red", lw=1.1)
    axa.set_title(f"(a) $K=2$ (oracle: {len(pool)} choose 2)")

    mx = bars(axb, res4, ["tab:blue", "tab:cyan", "0.6", "0.75"])
    xr = len(res4)                                   # random-distribution entry
    axb.errorbar([xr], [rand_vals.mean()], yerr=[rand_vals.std()], fmt="o",
                 color="tab:purple", capsize=5, ms=7)
    axb.plot([xr], [rand_vals.max()], "v", color="tab:purple", ms=7)
    axb.set_xticks(list(range(len(res4))) + [xr],
                   list(res4.keys()) + [f"random\n({rand_trials})"])
    axb.tick_params(axis="x", rotation=20)
    axb.set_ylim(top=mx * 1.12)
    axb.set_title(r"(b) $K=4$ ($\binom{25}{4}=12650$: not enumerated)")

    C.plot_grid(axc, coords, title="(c) selected sites ($K=4$)")
    C.mark_selected(axc, coords, res4["greedy+swap"][0], color="tab:blue",
                    label="greedy+swap")
    C.mark_selected(axc, coords, res4["recv-power"][0], color="tab:orange",
                    label="recv-power", marker="s")
    C.mark_selected(axc, coords, res4["distance"][0], color="tab:green",
                    label="distance", marker="^")
    axc.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    p = f"{C.OUT}/exp3_selection.pdf"; fig.savefig(p); print("saved", p)

    best_k4 = max(res4["greedy"][1], res4["greedy+swap"][1])
    ok = (res2["oracle"][1] >= max(v for _, v in res2.values()) - 0.05   # the roof
          and res2["greedy+swap"][1] >= res2["recv-power"][1] - 1e-6
          and best_k4 >= rand_vals.mean() - 1e-6)
    return ok


if __name__ == "__main__":
    print("exp3_selection: strategies vs oracle and baselines (5x5) ...")
    raise SystemExit(0 if main(selftest="--selftest" in sys.argv) else 1)
