"""Multi-pair NRG: relay selection under interference (via cmi-dag).

Two Tx/Rx pairs share the candidate relay grid. Each active relay forwards both
transmitters (shared medium), so it is double-edged -- it can boost the desired
links but also pour interference into the wrong receiver. We read each pair's rate
from the multi-root conditional MI (relay_grid_dag.pair_rates) and compare:

  direct-only | received-power (interference-blind) | greedy sum-rate | oracle

PASS if sum-rate-aware selection beats the interference-blind baseline and direct-
only, greedy is near the oracle, and the interference is real (free > TIN), with the
relay-borne component persisting even when the cross-direct links vanish.

Output: out/multipair_interference.png   (rates are bits/s/Hz, FD; HD = x1/2)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import relay_grid_dag as rgd

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
M, K = 2, 2
PAIRS = rgd.multipair.DEFAULT_PAIRS                      # two pairs offset in y


def main():
    s, names, coords = rgd.build_pair_scene(PAIRS)
    sr = lambda idx: rgd.weighted_sum_rate(s, M, [names[i] for i in idx])

    direct = sr([])
    rp_idx = rgd.received_power_pairs(s, M, names, K)
    g_idx = rgd.select_greedy_sumrate(s, M, names, K)
    o_idx, o_sr = rgd.select_exhaustive_sumrate(s, M, names, K)
    rp, gr = sr(rp_idx), sr(g_idx)

    rates_g = rgd.pair_rates(s, M, [names[i] for i in g_idx])
    cost = sum(rates_g["free"][u] - rates_g["tin"][u] for u in range(M))
    relay_borne = sum(v - t for v, t in zip(
        rgd.pair_rates(s, M, [names[i] for i in g_idx], cross_atten=1e-6)["free"],
        rgd.pair_rates(s, M, [names[i] for i in g_idx], cross_atten=1e-6)["tin"]))

    ok = (gr > rp and gr > direct and o_sr + 1e-6 >= gr and cost > 0.1
          and relay_borne > 0.1)

    # ---- figure: (left) scene + selections, (right) sum-rate bars -----------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.0))
    axL.scatter(coords[:, 0], coords[:, 1], s=70, c="0.75", edgecolors="k",
                linewidths=0.4, zorder=3, label="candidate relays")
    cols = ["tab:blue", "tab:green"]
    for u, (tx, rx) in enumerate(PAIRS):
        axL.plot([tx[0], rx[0]], [tx[1], rx[1]], "--", color=cols[u], alpha=0.5)
        axL.scatter(*tx, marker="s", s=130, c=cols[u], edgecolors="k", zorder=5,
                    label=f"Tx{u}/Rx{u}")
        axL.scatter(*rx, marker="D", s=130, c=cols[u], edgecolors="k", zorder=5)
    axL.scatter(coords[rp_idx, 0], coords[rp_idx, 1], s=300, marker="s",
                facecolors="none", edgecolors="darkorange", linewidths=2.4,
                zorder=6, label="received-power")
    axL.scatter(coords[g_idx, 0], coords[g_idx, 1], s=320, facecolors="none",
                edgecolors="red", linewidths=2.8, zorder=7, label="greedy sum-rate")
    axL.set_xlabel("x [wavelengths]"); axL.set_ylabel("y [wavelengths]")
    axL.set_aspect("equal", adjustable="datalim"); axL.grid(True, lw=0.3, alpha=0.5)
    axL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8,
               frameon=False)
    axL.set_title("2 interfering pairs: greedy (red) vs received-power (orange)")

    labels = ["direct\nonly", "received\npower", "greedy\nsum-rate", "oracle"]
    vals = [direct, rp, gr, o_sr]
    bars = axR.bar(labels, vals, color=["0.6", "darkorange", "red", "k"])
    axR.bar_label(bars, fmt="%.1f", fontsize=8)
    axR.set_ylabel("sum-rate Σ_u R_u  [bits/s/Hz, FD]")
    axR.set_title(f"interference cost (free−TIN) = {cost:.1f} bits "
                  f"(relay-borne {relay_borne:.1f})")
    axR.grid(True, axis="y", lw=0.3, alpha=0.5)

    fig.suptitle("Multi-pair NRG: interference-aware selection (cmi-dag CMI)",
                 fontsize=11)
    path = f"{OUT}/multipair_interference.png"
    fig.savefig(path, bbox_inches="tight")

    print("=== multi-pair NRG: interference-aware selection ===")
    print(f"M={M} pairs, L={len(names)} candidates, K={K}")
    print(f"greedy {g_idx}: per-pair TIN {[round(x,2) for x in rates_g['tin']]}, "
          f"free {[round(x,2) for x in rates_g['free']]}")
    print(f"sum-rate (TIN, FD):  direct {direct:.2f} | received-power {rp:.2f} "
          f"| greedy {gr:.2f} | oracle {o_sr:.2f}")
    print(f"interference cost = {cost:.2f} bits  (relay-borne {relay_borne:.2f} "
          f"persists at cross_atten->0)")
    print(f"saved {path}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
