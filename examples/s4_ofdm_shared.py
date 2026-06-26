"""S4 — OFDM: the shared-activation combinatorics (the wideband wrinkle)

In OFDM NF-PRG the precoder/relay gains may vary per subcarrier, but the physical
relay ON/OFF set a_l is *shared across the whole band*. Near-field beam squint
means each subcarrier would, on its own, prefer a different relay set; the shared
set must compromise. This sub-test makes that visible:

  * per-subcarrier oracle sum  = upper bound (each subcarrier picks freely)  -- not realisable
  * best SHARED K-set          = maximise the wideband sum-rate Sum_k I_k     -- realisable
  * received-power shared K-set = naive baseline

PASS if the realisable shared optimum sits strictly below the (unrealisable)
per-subcarrier upper bound (a real sharing cost), beats the received-power shared
set, and the per-subcarrier favourite sets are not all identical (squint is real).

Output: out/s4_ofdm_shared.png
"""
import itertools

import numpy as np

import _common as C

K = 2
F_C = 10.0
S = 5
DF = 1.0


def main():
    plt = C._mpl()
    s, names, coords = C.build_candidate_scene("near")
    k_waves = C.nfd.subcarrier_wavenumbers(F_C, S, DF)
    L = len(names)
    combos = list(itertools.combinations(range(L), K))

    # MI table: rows = subcarriers, cols = candidate K-subsets
    table = np.zeros((S, len(combos)))
    for si, kw in enumerate(k_waves):
        for ci, combo in enumerate(combos):
            table[si, ci] = C.subset_mi(s, [names[i] for i in combo], k_wave=kw)

    # per-subcarrier oracle (each subcarrier free to choose) -> upper bound
    per_sc_best_ci = table.argmax(1)
    per_sc_sets = [combos[c] for c in per_sc_best_ci]
    upper_bound = table.max(1).sum()

    # best shared set: maximise the band sum
    band_sum = table.sum(0)
    shared_ci = int(band_sum.argmax())
    shared_set = combos[shared_ci]
    shared_rate = band_sum[shared_ci]

    # received-power shared baseline (frequency-independent score)
    rp_idx = tuple(C.select_received_power(s, names, K))
    rp_ci = combos.index(tuple(sorted(rp_idx)))
    rp_rate = band_sum[rp_ci]

    sharing_cost = (upper_bound - shared_rate) / upper_bound
    squint = len({frozenset(x) for x in per_sc_sets}) > 1
    ok = (shared_rate < upper_bound - 1e-6 and shared_rate >= rp_rate - 1e-6
          and squint)

    # ---- figure: (left) selection map, (right) per-subcarrier rate bars
    fig = plt.figure(figsize=(12, 5.0))
    axm = fig.add_subplot(1, 2, 1)
    axb = fig.add_subplot(1, 2, 2)

    C.plot_grid(axm, coords, title="shared set (red) vs each subcarrier's favourite")
    cmap = plt.cm.plasma(np.linspace(0, 0.9, S))
    for si, st in enumerate(per_sc_sets):
        pts = coords[list(st)]
        axm.scatter(pts[:, 0], pts[:, 1], s=70, color=cmap[si], marker="x",
                    linewidths=2, zorder=6,
                    label=f"sc{si} best" if si in (0, S - 1) else None)
    C.mark_selected(axm, coords, shared_set, color="red",
                    label="shared optimum", marker="o")
    C.mark_selected(axm, coords, rp_idx, color="darkorange",
                    label="received-power", marker="s")
    axm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
               fontsize=8, frameon=False)

    # per-subcarrier rates: upper bound vs shared vs received-power
    x = np.arange(S)
    w = 0.27
    axb.bar(x - w, table.max(1), w, label="per-sc oracle (upper bnd)", color="0.6")
    axb.bar(x, table[:, shared_ci], w, label="shared optimum", color="red")
    axb.bar(x + w, table[:, rp_ci], w, label="received-power", color="darkorange")
    axb.set_xticks(x); axb.set_xticklabels([f"sc{i}" for i in range(S)])
    axb.set_ylabel("I_k [bits/s/Hz]")
    axb.set_title(f"band sum: upper={upper_bound:.1f}  "
                  f"shared={shared_rate:.1f}  rp={rp_rate:.1f}")
    axb.legend(fontsize=8); axb.grid(True, axis="y", lw=0.3, alpha=0.5)

    fig.suptitle("S4 — OFDM shared activation: one set for the whole band pays a squint cost",
                 fontsize=11)
    path = f"{C.OUT}/s4_ofdm_shared.png"
    fig.savefig(path, bbox_inches="tight")

    print("=== S4: OFDM shared activation ===")
    print(f"subcarriers S={S}, f_c={F_C}, df={DF}, K={K}")
    for si, st in enumerate(per_sc_sets):
        print(f"  sc{si}: best set {st}  I={table[si].max():.2f}")
    print(f"per-subcarrier oracle sum (upper bound) = {upper_bound:.3f}")
    print(f"best SHARED set {shared_set} -> band sum = {shared_rate:.3f}")
    print(f"received-power shared {tuple(sorted(rp_idx))} -> band sum = {rp_rate:.3f}")
    print(f"sharing cost vs upper bound = {sharing_cost*100:.1f}%")
    print(f"subcarriers disagree (squint) = {squint}")
    print(f"saved {path}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
