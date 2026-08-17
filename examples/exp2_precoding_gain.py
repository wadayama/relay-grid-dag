"""EXP2 -- what precoding buys: gain over scalar-AF vs K, and the ablation.

Canonical single-pair scene (5x5 grid). Two panels:
  (a) optimized MI f(S_k) vs the conventional scalar-AF relay along the greedy
      nested sets S_0 subset S_1 subset ... subset S_4 (S_0 = direct link). Checks
      f(empty) = water-filling on H_sd and monotonicity (free disposal) on the way.
  (b) ablation at the fixed S_3 set: scalar-AF / Tx-precoder-only (water-filling over
      the scalar-AF relay channel) / relay-precoder-only (F isotropic, engine with
      freeze_F) / joint -- which lever buys what.

    uv run --extra examples python examples/exp2_precoding_gain.py [--selftest]
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

import _common as C
import relay_grid_dag as rgd

DT = rgd.DTYPE
P_TX = P_RELAY = 100.0
SIG = 1.0
K_MAX = 4
K_EXT = 8                   # extended sweep for the conference companion figure
K_ABL = 3
SEL_ITERS = 100             # greedy-selection engine depth (warm-started)
EVAL_ITERS = 300            # reporting depth
RESTARTS = 2


def greedy_nested(scene, names, K, *, iters, restarts=1):
    """Forward greedy keeping the ADD ORDER (so every prefix is the greedy set),
    warm-started like the library's select_greedy."""
    order, remaining = [], set(range(len(names)))
    F_star, W_star = None, []
    for _ in range(K):
        best = None
        for j in remaining:
            subset = [names[i] for i in order + [j]]
            kw = dict(iters=iters, restarts=restarts)
            if F_star is not None:
                kw.update(F_init=F_star, W_init=W_star + [None])
            cap, F, Ws = rgd.optimize_precoders(scene, "tx", subset, "rx", **kw)
            if best is None or cap > best[0]:
                best = (cap, j, F, Ws)
        order.append(best[1]); remaining.discard(best[1])
        F_star, W_star = best[2], best[3]
    return order


def txonly_se(scene, sel):
    """Optimal Tx precoder over the conventional scalar-AF relay channel
    (water-filling; relay gains fixed at the isotropic operating point)."""
    n_tx, n_rx = scene.n_ant("tx"), scene.n_ant("rx")
    Q0 = P_TX / n_tx
    Heff = scene.channel("tx", "rx", atten=scene.direct_atten).to(DT)
    Cov = (SIG ** 2) * torch.eye(n_rx, dtype=DT)
    for nm in sel:
        H_sr = scene.channel("tx", nm).to(DT)
        H_rd = scene.channel(nm, "rx").to(DT)
        K_ll = Q0 * H_sr @ H_sr.mH + (SIG ** 2) * torch.eye(H_sr.shape[0], dtype=DT)
        g = math.sqrt(P_RELAY / float(torch.trace(K_ll).real))
        Heff = Heff + g * (H_rd @ H_sr)
        Cov = Cov + (g ** 2) * (SIG ** 2) * (H_rd @ H_rd.mH)
    return rgd.waterfilling_capacity(Heff, Cov, P_TX)[0]


def main(selftest=False):
    plt = C._mpl()
    scene, names, coords = C.canonical_scene()
    sel_iters = 15 if selftest else SEL_ITERS
    eval_iters = 40 if selftest else EVAL_ITERS
    restarts = 1 if selftest else RESTARTS

    # ---- (a) gain vs K along the greedy nested sets --------------------------
    # Greedy is prefix-consistent, so one run to k_ext yields every S_k; the
    # original figure uses the K<=K_MAX slice (unchanged), the conference
    # companion extends the sweep to K_EXT.
    k_ext = K_MAX if selftest else K_EXT
    order = greedy_nested(scene, names, k_ext, iters=sel_iters)
    print(f"greedy add order: {order}  (S_k = first k)")
    Ks, base, opt = [], [], []
    for k in range(k_ext + 1):
        subset = [names[i] for i in sorted(order[:k])]
        Ks.append(k)
        base.append(rgd.subset_mi(scene, subset))
        opt.append(rgd.optimized_mi(scene, "tx", subset, "rx",
                                    iters=eval_iters, restarts=restarts))
        print(f"  K={k}: scalar-AF={base[-1]:6.2f}  optimized={opt[-1]:6.2f}  "
              f"gain={opt[-1] - base[-1]:+6.2f} bits/s/Hz")

    # ties: f(empty) = water-filling on the direct channel; monotone in K
    H_sd = scene.channel("tx", "rx", atten=scene.direct_atten).to(DT)
    wf = rgd.waterfilling_capacity(H_sd, (SIG ** 2) * torch.eye(scene.n_ant("rx"), dtype=DT),
                                   P_TX)[0]
    print(f"tie: f(empty)={opt[0]:.4f} vs direct water-filling={wf:.4f} "
          f"(gap {abs(opt[0] - wf):.1e})")
    mono = all(opt[i + 1] >= opt[i] - 1e-3 for i in range(k_ext))
    print(f"tie: monotone in K: {mono}")

    # ---- (b) ablation at the fixed S_{K_ABL} ---------------------------------
    sel = [names[i] for i in sorted(order[:K_ABL])]
    n_tx = scene.n_ant("tx")
    F_iso = math.sqrt(P_TX / n_tx) * torch.eye(n_tx, dtype=DT)
    abl = {
        "scalar-AF": rgd.subset_mi(scene, sel),
        "Tx only": txonly_se(scene, sel),
        "relay only": rgd.optimize_precoders(scene, "tx", sel, "rx", iters=eval_iters,
                                             restarts=restarts, F_init=F_iso,
                                             freeze_F=True)[0],
        "joint": rgd.optimize_precoders(scene, "tx", sel, "rx", iters=eval_iters,
                                        restarts=restarts)[0],
    }
    for k, v in abl.items():
        print(f"  ablation {k:10s} = {v:6.2f} bits/s/Hz")

    # ---- figure ---------------------------------------------------------------
    def draw_gain(ax, kmax, title):
        sl = slice(0, kmax + 1)
        ax.plot(Ks[sl], base[sl], "o--", color="0.5",
                label="conventional scalar-AF")
        ax.plot(Ks[sl], opt[sl], "s-", color="tab:blue",
                label="optimized MI $f(S_K)$")
        ax.plot([0], [wf], "*", color="tab:red", ms=13, zorder=6,
                label="direct water-filling")
        ax.set_xlabel("active relays $K$ (greedy nested sets)", fontsize=13)
        ax.set_ylabel("SE [bits/s/Hz]", fontsize=13)
        ax.set_xticks(Ks[sl]); ax.tick_params(labelsize=11)
        ax.legend(fontsize=11); ax.grid(alpha=0.3)
        if title:
            ax.set_title(title)

    def gain_ablation_figure(kmax, outfile):
        """The two-panel figure with panel (a) truncated at ``kmax``."""
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.6, 4.2),
                                       gridspec_kw={"width_ratios": [1.25, 1.0]})
        draw_gain(axa, kmax, "(a) precoding gain vs $K$")
        labels = list(abl.keys()); vals = [abl[k] for k in labels]
        colors = ["0.6", "tab:orange", "tab:green", "tab:blue"]
        bars = axb.bar(labels, vals, color=colors)
        for b, v in zip(bars, vals):
            axb.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                     ha="center", fontsize=10)
        axb.set_ylabel("SE [bits/s/Hz]"); axb.grid(axis="y", alpha=0.3)
        axb.set_title(f"(b) ablation at the $K={K_ABL}$ set")
        fig.tight_layout()
        p = f"{C.OUT}/{outfile}"; fig.savefig(p); print("saved", p)

    gain_ablation_figure(K_MAX, "exp2_precoding_gain.pdf")
    if k_ext > K_MAX:
        gain_ablation_figure(k_ext, f"exp2_precoding_gain_k{k_ext}.pdf")
        # single-panel gain-only variant (conference layout: the ablation is
        # reported in the text, so panel (a) alone fits a single column)
        figs, axs = plt.subplots(figsize=(6.4, 4.4))
        draw_gain(axs, k_ext, "")
        figs.tight_layout()
        ps = f"{C.OUT}/exp2_gain_k{k_ext}_single.pdf"
        figs.savefig(ps); print("saved", ps)

    wf_tol = 0.5 if selftest else 5e-2          # short PGA can't fully close the tie
    ok = (mono and abs(opt[0] - wf) < wf_tol
          and all(o >= b - 1e-6 for o, b in zip(opt, base))
          and abl["joint"] >= max(abl["Tx only"], abl["relay only"]) - 5e-2)
    return ok


if __name__ == "__main__":
    print("exp2_precoding_gain: gain vs K + ablation ...")
    raise SystemExit(0 if main(selftest="--selftest" in sys.argv) else 1)
