"""EXP5 -- relay handover under receiver mobility (single pair, applied).

Canonical 5x5 scene, K=3. The Rx moves along x=20, y in [-5, 5] (13 frames); the
grid is fixed, so adapting means *re-selecting* which K relays are active -- a
discrete relay handover. Per frame we compare:
  * adaptive : greedy re-selection at the current Rx position;
  * static   : the frame-0 set kept, precoders re-optimized only;
  * direct   : the direct link alone.
Panels: geometry (trajectory + relays coloured by when they serve) | handover map
(active relays per frame) | SE per frame.

    uv run --extra examples python examples/exp5_handover.py [--selftest]
"""
from __future__ import annotations

import sys

import numpy as np

import _common as C
import relay_grid_dag as rgd

K = 3
RX_X = 20.0
YS = np.linspace(-5.0, 5.0, 13)
SEL_ITERS = 25
EVAL_ITERS = 250
RESTARTS = 2


def main(selftest=False):
    plt = C._mpl()
    scene, names, coords = C.canonical_scene()
    ys = YS[::6] if selftest else YS
    sel_iters = 6 if selftest else SEL_ITERS
    eval_iters = 30 if selftest else EVAL_ITERS
    restarts = 1 if selftest else RESTARTS
    L = len(names)

    idx_per, se_a, se_s, se_d = [], [], [], []
    static_set = None
    for f, y in enumerate(ys):
        scene.move("rx", [RX_X, float(y)])
        a_idx = rgd.select_greedy(scene, names, K, iters=sel_iters, restarts=restarts)
        if f == 0:
            static_set = a_idx
        se_a.append(rgd.optimize_precoders(scene, "tx", [names[i] for i in a_idx], "rx",
                                           iters=eval_iters, restarts=restarts)[0])
        se_s.append(rgd.optimize_precoders(scene, "tx", [names[i] for i in static_set],
                                           "rx", iters=eval_iters, restarts=restarts)[0])
        se_d.append(rgd.direct_only_mi(scene))
        idx_per.append(a_idx)
        print(f"  frame {f:2d}  Rx=({RX_X},{y:+.1f})  set={a_idx}  "
              f"SE adapt={se_a[-1]:.2f} static={se_s[-1]:.2f} direct={se_d[-1]:.2f}")
    scene.move("rx", [RX_X, 0.0])

    sel = np.zeros((len(ys), L), bool)
    for f, idx in enumerate(idx_per):
        sel[f, idx] = True
    used = sel.any(axis=0)
    mean_frame = np.array([sel[:, i].nonzero()[0].mean() if used[i] else np.nan
                           for i in range(L)])

    fig, (axg, axh, axe) = plt.subplots(1, 3, figsize=(15.6, 4.4))
    # --- geometry ---
    axg.scatter(coords[:, 0], coords[:, 1], s=60, c="0.85", edgecolors="k",
                linewidths=0.4, zorder=2)
    sc = axg.scatter([RX_X] * len(ys), ys, c=range(len(ys)), cmap="viridis",
                     s=55, zorder=4, label="Rx trajectory")
    ring_c = plt.cm.viridis(mean_frame[used] / max(len(ys) - 1, 1))
    axg.scatter(coords[used, 0], coords[used, 1], s=300, facecolors="none",
                edgecolors=ring_c, linewidths=2.6, zorder=5)
    axg.scatter([0.0], [0.0], marker="s", s=120, c="tab:red", edgecolors="k",
                zorder=6, label="Tx")
    axg.set_title("(a) geometry: Rx path, relays used\n(ring colour = mean serving frame)")
    axg.set_xlabel(r"x [$\lambda$]"); axg.set_ylabel(r"y [$\lambda$]")
    axg.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=axg, fraction=0.046, label="frame")
    # --- handover map ---
    for f, idx in enumerate(idx_per):
        axh.scatter([f] * len(idx), idx, c=[f] * len(idx), cmap="viridis",
                    vmin=0, vmax=len(ys) - 1, s=70)
    axh.set_xlabel("frame"); axh.set_ylabel("candidate relay index")
    axh.set_title("(b) handover map: active set per frame")
    axh.set_yticks(range(0, L, 4)); axh.grid(alpha=0.25)
    # --- SE ---
    fr = range(len(ys))
    axe.plot(fr, se_a, "o-", color="tab:blue", label="adaptive (re-select)")
    axe.plot(fr, se_s, "s--", color="tab:orange", label="static (frame-0 set)")
    axe.plot(fr, se_d, "^:", color="0.4", label="direct only")
    axe.set_xlabel("frame"); axe.set_ylabel("SE [bits/s/Hz]")
    axe.set_title("(c) spectral efficiency per frame")
    axe.legend(fontsize=9); axe.grid(alpha=0.3)
    fig.tight_layout()
    p = f"{C.OUT}/exp5_handover.pdf"; fig.savefig(p); print("saved", p)

    gain = float(np.mean(np.array(se_a) - np.array(se_s)))
    n_ho = sum(1 for a, b in zip(idx_per, idx_per[1:]) if a != b)
    print(f"mean adaptive-over-static gain = {gain:+.2f} bits/s/Hz; "
          f"handovers = {n_ho}/{len(ys) - 1} frame transitions")
    adapt_tol = 2.0 if selftest else 1e-6      # shallow selftest search is noisy
    return bool(np.all(np.isfinite(se_a))
                and np.mean(se_a) >= np.mean(se_s) - adapt_tol
                and np.mean(se_a) >= np.mean(se_d))


if __name__ == "__main__":
    print("exp5_handover: relay handover under Rx mobility ...")
    raise SystemExit(0 if main(selftest="--selftest" in sys.argv) else 1)
