"""S2b — Robustness sweep: is the value of clever selection systematic?

S2 showed, for one geometry, that MI-aware selection beats naive selection. Here
we turn that anecdote into a distribution: over many randomised geometries, several
K, and two direct-attenuation levels, we record the MI gaps between strategies and
add a distance-based baseline. We report the spread (median / IQR), not a single
number.

PASS if, across the sweep, the oracle beats received-power AND distance-based by a
positive median margin, greedy stays within a small median gap of the oracle (a
cheap near-oracle), and naive received-power is frequently at or below the random
mean (naive magnitude-only selection misleads).

Output: out/s2b_robustness.png
"""
import itertools
import numpy as np

import _common as C

N_GEO = 10            # random geometries
KS = (2, 3, 4)
ATTENS = (0.3, 0.7)
L_NX, L_NY = 4, 5     # 20 candidates -> oracle stays cheap (C(20,4)=4845)
JITTER = 0.6          # wavelength std of per-point jitter


def random_scene(seed, direct_atten):
    rng = np.random.default_rng(seed)
    base = C.grid_coords(L_NX, L_NY)
    coords = base + rng.normal(0, JITTER, base.shape)
    s = C.nfd.Scene(model="near", direct_atten=direct_atten)
    s.add_node("tx", list(C.TX_XY), C.N_TX, "source")
    # jitter Rx a little too, to vary the geometry
    rx = [20.0 + float(rng.uniform(-1.5, 1.5)), float(rng.uniform(-2, 2))]
    s.add_node("rx", rx, C.N_RX, "sink")
    names = []
    for i, (x, y) in enumerate(coords):
        s.add_node(f"c{i}", [float(x), float(y)], C.N_RELAY, "relay")
        names.append(f"c{i}")
    return s, names, coords, rx


def one_config(seed, K, atten):
    s, names, coords, rx = random_scene(seed, atten)
    direct = C.direct_only_mi(s)
    ora_idx, ora = C.select_exhaustive(s, names, K)
    gr = C.subset_mi(s, [names[i] for i in C.select_greedy_mi(s, names, K)])
    rp = C.subset_mi(s, [names[i] for i in C.select_received_power(s, names, K)])
    di = C.subset_mi(s, [names[i] for i in C.select_distance(coords, K, rx=rx)])
    rnd = C.random_subset_stats(s, names, K, trials=150, seed=seed + 99)
    return dict(direct=direct, oracle=ora, greedy=gr, recvpow=rp, distance=di,
                rnd_mean=rnd["mean"], rnd_max=rnd["max"])


def main():
    plt = C._mpl()
    rows = []
    for gi in range(N_GEO):
        for K in KS:
            for atten in ATTENS:
                r = one_config(1000 * gi + K, K, atten)
                r.update(geo=gi, K=K, atten=atten)
                rows.append(r)

    arr = lambda key: np.array([r[key] for r in rows])
    gap_rp = arr("oracle") - arr("recvpow")
    gap_di = arr("oracle") - arr("distance")
    gap_rnd = arr("oracle") - arr("rnd_mean")
    gap_greedy = arr("oracle") - arr("greedy")
    rp_minus_rnd = arr("recvpow") - arr("rnd_mean")
    di_minus_rnd = arr("distance") - arr("rnd_mean")

    def stat(x):
        return np.median(x), np.percentile(x, 25), np.percentile(x, 75)

    frac_rp_below_rnd = float(np.mean(rp_minus_rnd <= 0))

    med_rp = stat(gap_rp)[0]
    med_di = stat(gap_di)[0]
    med_greedy = stat(gap_greedy)[0]
    ok = (med_rp >= 0.5 and med_di >= 0.5 and med_greedy <= 0.3
          and frac_rp_below_rnd >= 0.4)

    # ---- figure: (left) gap distributions vs oracle, (right) naive-vs-random
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.0))
    data = [gap_greedy, gap_rnd, gap_rp, gap_di]
    labels = ["oracle\n- greedy", "oracle\n- random", "oracle\n- recvpow",
              "oracle\n- distance"]
    bp = axL.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["tab:green", "0.7", "tab:red", "tab:purple"]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    axL.axhline(0, color="k", lw=0.8, ls="--")
    axL.set_ylabel("MI gap [bits/s/Hz]")
    axL.set_title(f"MI gap vs oracle over {len(rows)} configs "
                  f"({N_GEO} geom x K{KS} x atten{ATTENS})")
    axL.grid(True, axis="y", lw=0.3, alpha=0.5)

    bp2 = axR.boxplot([rp_minus_rnd, di_minus_rnd],
                      tick_labels=["recvpow\n- random", "distance\n- random"],
                      showmeans=True, patch_artist=True)
    for patch, col in zip(bp2["boxes"], ["tab:red", "tab:purple"]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    axR.axhline(0, color="k", lw=0.8, ls="--")
    axR.set_ylabel("MI gap [bits/s/Hz]")
    axR.set_title(f"naive vs random  (recvpow <= random in {frac_rp_below_rnd*100:.0f}% of configs)")
    axR.grid(True, axis="y", lw=0.3, alpha=0.5)

    fig.suptitle("S2b — value of clever selection is systematic, not anecdotal",
                 fontsize=11)
    path = f"{C.OUT}/s2b_robustness.png"
    fig.savefig(path, bbox_inches="tight")

    print("=== S2b: robustness sweep ===")
    print(f"configs: {len(rows)}  ({N_GEO} geometries x K{KS} x atten{ATTENS})")
    print("gap vs oracle [median (IQR)]:")
    for nm, g in [("oracle-greedy", gap_greedy), ("oracle-random", gap_rnd),
                  ("oracle-recvpow", gap_rp), ("oracle-distance", gap_di)]:
        m, lo, hi = stat(g)
        print(f"  {nm:16s}: {m:6.2f}  ({lo:.2f} .. {hi:.2f})")
    print(f"received-power <= random mean in {frac_rp_below_rnd*100:.0f}% of configs")
    print(f"distance     [median recvpow-rnd]={np.median(rp_minus_rnd):.2f}, "
          f"[median dist-rnd]={np.median(di_minus_rnd):.2f}")
    print(f"saved {path}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
