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
import json
import os
import sys

import numpy as np
import torch

import _common as C
import relay_grid_dag as rgd
from relay_grid_dag import viz

SEARCH_ITERS = 120           # engine depth during the combinatorial search
EVAL_ITERS = 300             # common reporting depth (also used for random subsets)
EVAL_RESTARTS = 2
RAND_TRIALS = 30
SEED = 0
K4 = 4
ORACLE4_JSON = f"{C.OUT}/exp3_oracle_k4.json"
ORACLE4_NPZ = f"{C.OUT}/exp3_oracle_k4_values.npz"


def oracle_k4():
    """Exhaustive K=4 oracle at the FULL reporting depth (C(25,4)=12,650 inner
    optimizations, hours of CPU). Checkpoints every 200 subsets so an
    interrupted run resumes; results are cached for the figure build."""
    scene, names, _ = C.canonical_scene()
    combos = list(itertools.combinations(range(len(names)), K4))
    n = len(combos)
    vals, start = np.full(n, np.nan), 0
    if os.path.exists(ORACLE4_NPZ):
        d = np.load(ORACLE4_NPZ)
        vals, start = d["vals"], int(d["count"])
        print(f"oracle-k4: resuming at {start}/{n}")
    for i in range(start, n):
        idx = list(combos[i])
        vals[i] = rgd.optimized_mi(scene, "tx", [names[j] for j in idx], "rx",
                                   iters=EVAL_ITERS, restarts=EVAL_RESTARTS)
        if (i + 1) % 200 == 0 or i == n - 1:
            np.savez(ORACLE4_NPZ, vals=vals, count=i + 1)
            print(f"  oracle-k4: {i + 1}/{n}  best so far "
                  f"{np.nanmax(vals):.2f} bits/s/Hz", flush=True)
    best_i = int(np.nanargmax(vals))
    res = {"best_idx": sorted(combos[best_i]), "best_mi": float(vals[best_i]),
           "n": n, "eval_iters": EVAL_ITERS, "eval_restarts": EVAL_RESTARTS}
    with open(ORACLE4_JSON, "w") as f:
        json.dump(res, f, indent=1)
    print(f"oracle-k4: best set {res['best_idx']}  "
          f"f(S)={res['best_mi']:.2f} bits/s/Hz  (over {n} subsets)")
    return res


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

    # ---- companion figures (conference layout) -------------------------------
    # Two single-column figures from the same data. Bars: the K=4 strategies
    # (the exhaustive K=4 oracle bar appears when its cached --oracle-k4 result
    # is present). Fields: the carrier field of the direct-only, greedy K=2,
    # and greedy K=4 configurations, rendered from the SAME reporting-depth
    # precoders that produce the reported MI (consistency: SPEC.md Sec. 6 /
    # tie-point T1) on a SHARED 0-dB reference, so the three panels are
    # directly comparable.
    res_a: dict = {}
    if os.path.exists(ORACLE4_JSON):
        with open(ORACLE4_JSON) as f:
            o4 = json.load(f)
        res_a["oracle"] = (o4["best_idx"], o4["best_mi"])
        print(f"  bars figure includes cached K=4 oracle: {o4['best_idx']} "
              f"f(S)={o4['best_mi']:.2f}")
    for k in ("greedy", "recv-power", "distance"):   # no swap in this layout
        res_a[k] = res4[k]
    col = {"oracle": "tab:red", "greedy": "tab:blue",
           "recv-power": "0.6", "distance": "0.75"}
    figb, axA = plt.subplots(figsize=(6.4, 4.2))
    labels_a = list(res_a.keys())
    xa = np.arange(len(labels_a) + 1, dtype=float)
    va = [res_a[k][1] for k in labels_a]
    axA.bar(xa[:-1], va, color=[col[k] for k in labels_a])
    for x, v in zip(xa[:-1], va):
        axA.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    axA.errorbar([xa[-1]], [rand_vals.mean()], yerr=[rand_vals.std()], fmt="o",
                 color="tab:purple", capsize=5, ms=6)
    axA.plot([xa[-1]], [rand_vals.max()], "v", color="tab:purple", ms=6)
    axA.set_xticks(xa, labels_a + [f"random\n({rand_trials})"])
    axA.tick_params(axis="x", rotation=20, labelsize=9)
    axA.set_ylim(0, max(va) * 1.14)
    axA.set_ylabel("optimized MI $f(S)$ [bits/s/Hz]")
    axA.grid(axis="y", alpha=0.3)
    figb.tight_layout()
    pb = f"{C.OUT}/exp3_bars.pdf"; figb.savefig(pb); print("saved", pb)

    # field progression: direct only -> greedy K=2 -> greedy K=4
    cap0, F_0, _ = rgd.optimize_precoders(scene, "tx", [], "rx",
                                          iters=eval_iters,
                                          restarts=eval_restarts)
    cap2, F_2, W_2 = rgd.optimize_precoders(
        scene, "tx", [names[i] for i in g2], "rx",
        iters=eval_iters, restarts=eval_restarts)
    cap4, F_4, W_4 = rgd.optimize_precoders(
        scene, "tx", [names[i] for i in g4], "rx",
        iters=eval_iters, restarts=eval_restarts)
    nx, ny = 220, 120
    xs = np.linspace(-2.5, 24.0, nx); ys = np.linspace(-6.5, 6.5, ny)
    gx, gy = np.meshgrid(xs, ys)
    probes = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1),
                          dtype=rgd.RDTYPE)
    ints = [
        viz.field_intensity(scene, "tx", probes, F=F_0).numpy(),
        viz.field_intensity(scene, "tx", probes, F=F_2,
                            relays=[names[i] for i in g2],
                            relay_mats=W_2).numpy(),
        viz.field_intensity(scene, "tx", probes, F=F_4,
                            relays=[names[i] for i in g4],
                            relay_mats=W_4).numpy(),
    ]
    ref = max(v.max() for v in ints)                 # shared 0-dB reference
    imgs = [np.clip(10 * np.log10(v / ref + 1e-12), -40, 0).reshape(ny, nx)
            for v in ints]
    panels = ((imgs[0], [], cap0, "(a) direct only"),
              (imgs[1], g2, cap2, "(b) greedy $K=2$"),
              (imgs[2], g4, cap4, "(c) greedy $K=4$"))
    figf, axes = plt.subplots(3, 1, figsize=(6.4, 9.0), sharex=True)
    for ax, (img, idx, cap, ttl) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]),
                       cmap="viridis", vmin=-40.0, vmax=0.0, aspect="equal")
        ax.scatter(coords[:, 0], coords[:, 1], s=22, facecolors="none",
                   edgecolors="0.8", linewidths=0.5, zorder=4)
        if idx:
            sel = coords[list(idx)]
            ax.scatter(sel[:, 0], sel[:, 1], s=70, facecolors="none",
                       edgecolors="#ff2bd6", linewidths=1.8, zorder=5)
        ax.plot([0], [0], "s", color="red", ms=8, zorder=6)
        ax.plot([20], [0], "D", color="cyan", ms=8, zorder=6)
        ax.set_title(f"{ttl} ($f(S)={cap:.1f}$)", fontsize=12)
        ax.set_ylabel(r"y [$\lambda$]", fontsize=13)
        ax.tick_params(labelsize=11)
    axes[-1].set_xlabel(r"x [$\lambda$]", fontsize=13)
    cb = figf.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    cb.set_label("field intensity [dB]", fontsize=12)
    cb.ax.tick_params(labelsize=11)
    pf = f"{C.OUT}/exp3_field_k.pdf"; figf.savefig(pf, bbox_inches="tight",
                                                   pad_inches=0.05)
    print("saved", pf)

    # ---- receiver-mobility field figure (conference layout) ------------------
    # Three Rx positions on the x=20 line; for EACH position the K=3 active set
    # is re-selected by warm-started greedy and the field is rendered from the
    # reporting-depth precoders (same consistency rule, shared 0-dB reference):
    # the active set and the field follow the moving receiver.
    K3 = 3
    rx_ys = (4.0, 0.0, -4.0)
    mob = []                              # (ry, idx, cap, intensity, ..., mimap)
    nx2, ny2 = (60, 34) if selftest else (180, 100)
    xs2 = np.linspace(-2.5, 24.0, nx2); ys2 = np.linspace(-6.5, 6.5, ny2)
    gx2, gy2 = np.meshgrid(xs2, ys2)
    for ry in rx_ys:
        s_r, names_r, _ = rgd.build_candidate_scene(rx_xy=(20.0, ry))
        idx_r = rgd.select_greedy(s_r, names_r, K3, iters=search_iters)
        subset_r = [names_r[i] for i in idx_r]
        cap_r, F_r, W_r = rgd.optimize_precoders(
            s_r, "tx", subset_r, "rx",
            iters=eval_iters, restarts=eval_restarts)
        i_r = viz.field_intensity(s_r, "tx", probes, F=F_r,
                                  relays=subset_r, relay_mats=W_r).numpy()
        # exact per-pixel array MI map: move the Rx array to each probe point
        # and evaluate the merge MI (9) under the FIXED precoders (F_r, W_r)
        mimap = np.zeros(nx2 * ny2)
        with torch.no_grad():
            for j, (px, py) in enumerate(zip(gx2.ravel(), gy2.ravel())):
                s_r.move("rx", [float(px), float(py)])
                spec = rgd.multirelay_merge(s_r, "tx", subset_r, "rx",
                                            sigma_r=1.0, sigma_d=1.0,
                                            precoder=F_r, relay_mats=W_r)
                mimap[j] = float(rgd.mi(spec))
        s_r.move("rx", [20.0, ry])
        mob.append((ry, idx_r, cap_r, i_r, mimap.reshape(ny2, nx2)))
        print(f"  mobility Rx=(20,{ry:+.0f}): greedy K=3 set={idx_r}  "
              f"f(S)={cap_r:6.2f} bits/s/Hz  "
              f"mimap max={mimap.max():.2f}")
    ref_m = max(v.max() for _, _, _, v, _ in mob)    # shared 0-dB reference
    figm, axesm = plt.subplots(3, 1, figsize=(6.4, 9.0), sharex=True)
    for ax, (ry, idx, cap, inten, _) in zip(axesm, mob):
        img = np.clip(10 * np.log10(inten / ref_m + 1e-12),
                      -40, 0).reshape(ny, nx)
        im = ax.imshow(img, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]),
                       cmap="viridis", vmin=-40.0, vmax=0.0, aspect="equal")
        ax.scatter(coords[:, 0], coords[:, 1], s=22, facecolors="none",
                   edgecolors="0.8", linewidths=0.5, zorder=4)
        sel = coords[list(idx)]
        ax.scatter(sel[:, 0], sel[:, 1], s=70, facecolors="none",
                   edgecolors="#ff2bd6", linewidths=1.8, zorder=5)
        ax.plot([0], [0], "s", color="red", ms=8, zorder=6)
        ax.plot([20], [ry], "D", color="cyan", ms=9, zorder=6)
        lab = chr(ord("a") + list(axesm).index(ax))
        ry_s = f"{ry:+.0f}" if ry else "0"
        ax.set_title(f"({lab}) Rx at $(20,{ry_s})$ ($f(S)={cap:.1f}$)",
                     fontsize=12)
        ax.set_ylabel(r"y [$\lambda$]", fontsize=13)
        ax.tick_params(labelsize=11)
    axesm[-1].set_xlabel(r"x [$\lambda$]", fontsize=13)
    cbm = figm.colorbar(im, ax=axesm, fraction=0.03, pad=0.03)
    cbm.set_label("field intensity [dB]", fontsize=12)
    cbm.ax.tick_params(labelsize=11)
    pm = f"{C.OUT}/exp3_field_rx.pdf"; figm.savefig(pm, bbox_inches="tight",
                                                    pad_inches=0.05)
    print("saved", pm)

    # MI-map rendering of the same three configurations: the exact rate a
    # receive array would attain at each point. The shared colour scale is
    # clipped at the best achieved f(S): saturated colour marks the region
    # where the probe rate meets or exceeds the reported rate (probes next
    # to the Tx or a radiating relay far exceed it and saturate).
    vmax_mi = max(cap for _, _, cap, _, _ in mob)
    figq, axesq = plt.subplots(3, 1, figsize=(6.4, 8.4), sharex=True)
    for ax, (ry, idx, cap, _, mimap) in zip(axesq, mob):
        im = ax.imshow(mimap, origin="lower",
                       extent=(xs2[0], xs2[-1], ys2[0], ys2[-1]),
                       cmap="viridis", vmin=0.0, vmax=vmax_mi, aspect="equal")
        ax.scatter(coords[:, 0], coords[:, 1], s=22, facecolors="none",
                   edgecolors="0.25", linewidths=0.7, zorder=4)
        sel = coords[list(idx)]
        ax.scatter(sel[:, 0], sel[:, 1], s=70, facecolors="none",
                   edgecolors="#ff2bd6", linewidths=1.8, zorder=5)
        ax.plot([0], [0], "s", color="red", ms=8, zorder=6)
        ax.plot([20], [ry], "D", color="cyan", ms=9, zorder=6)
        lab = chr(ord("a") + list(axesq).index(ax))
        ry_s = f"{ry:+.0f}" if ry else "0"
        ax.set_title(f"({lab}) Rx at $(20,{ry_s})$, "
                     f"SE$={cap:.1f}$ bits/s/Hz", fontsize=12)
        ax.set_ylabel(r"y [$\lambda$]", fontsize=13)
        ax.tick_params(labelsize=11)
    axesq[-1].set_xlabel(r"x [$\lambda$]", fontsize=13)
    cbq = figq.colorbar(im, ax=axesq, fraction=0.03, pad=0.03)
    cbq.set_label(r"$I(X;Y_p)$ [bits/s/Hz]", fontsize=12)
    cbq.ax.tick_params(labelsize=11)
    pq = f"{C.OUT}/exp3_mimap_rx.pdf"; figq.savefig(pq, bbox_inches="tight",
                                                    pad_inches=0.05)
    print("saved", pq)

    best_k4 = max(res4["greedy"][1], res4["greedy+swap"][1])
    ok = (res2["oracle"][1] >= max(v for _, v in res2.values()) - 0.05   # the roof
          and res2["greedy+swap"][1] >= res2["recv-power"][1] - 1e-6
          and best_k4 >= rand_vals.mean() - 1e-6)
    return ok


if __name__ == "__main__":
    if "--oracle-k4" in sys.argv:
        print("exp3_selection: exhaustive K=4 oracle at reporting depth ...")
        oracle_k4()
        raise SystemExit(0)
    print("exp3_selection: strategies vs oracle and baselines (5x5) ...")
    raise SystemExit(0 if main(selftest="--selftest" in sys.argv) else 1)
