"""EXP1 -- the differentiable engine at work: convergence/init + gradient efficiency.

Two panels on the canonical single-pair scene (SPEC.md Sec. 10, 5x5 grid):
  (a) convergence & initialization: best-so-far MI vs PGA iteration for independent
      random starts on the greedy K=3 active set. Substantiates the random-init +
      multi-start choice: a single random start is already competitive, the best-of-R
      envelope adds a little.
  (b) gradient efficiency: wall-clock of ONE complex reverse-mode AD sweep (returns
      the exact Wirtinger gradient w.r.t. F and ALL active {What_l} at once) vs the
      number of active relays, against a forward evaluation and the same gradient by
      central finite differences (cost ~ #real parameters x 2 forwards). The
      cheap-gradient principle, measured.

Also prints the AD-vs-FD max relative gradient error at K=1 (reproducibility number).

    uv run --extra examples python examples/exp1_engine.py [--selftest]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

import _common as C
import relay_grid_dag as rgd
from relay_grid_dag.precoding import _merge_mi, _build_init, _proj_ball

P_TX = P_RELAY = 100.0
SIG = 1.0
K_CONV = 3                  # active-set size for the convergence panel
SEL_ITERS = 100             # greedy-selection engine depth
CONV_ITERS = 400
N_STARTS = 8
TIME_KS = [1, 2, 4, 8, 16, 25]
REPS = 5                    # timing repetitions (median)


def _mi(scene, subset, F, What):
    return _merge_mi(scene, "tx", subset, "rx", F, What,
                     sigma_r=SIG, sigma_d=SIG, k_wave=rgd.K_WAVE)


def _leaves(scene, subset, seed=0):
    g = torch.Generator().manual_seed(seed)
    return _build_init(scene, "tx", subset, P_tx=P_TX, P_relay=P_RELAY, sigma_r=SIG,
                       k_wave=rgd.K_WAVE, d=scene.n_ant("tx"),
                       F_init=None, W_init=None, g=g)


def pga_trace(scene, subset, *, iters, step=0.05, step_min=1e-8, seed=0):
    """One PGA run (mirrors the engine's adaptive-step loop) recording the
    best-so-far MI after every iteration. Returns an (iters+1,) array."""
    F, What = _leaves(scene, subset, seed)

    def value(F_, W_):
        with torch.no_grad():
            return float(_mi(scene, subset, F_, W_))

    best = value(F, What)
    trace = [best]
    for _ in range(iters):
        F_l = F.detach().requires_grad_(True)
        W_l = [w.detach().requires_grad_(True) for w in What]
        I = _mi(scene, subset, F_l, W_l)
        I_cur = float(I.detach())
        grads = torch.autograd.grad(I, [F_l] + W_l)
        with torch.no_grad():
            F_new = _proj_ball(F + step * grads[0], P_TX)
            W_new = [_proj_ball(w + step * gg, P_RELAY)
                     for w, gg in zip(What, grads[1:])]
        I_new = value(F_new, W_new)
        if I_new >= I_cur:
            F, What = F_new, W_new
            step *= 1.1
            best = max(best, I_new)
        else:
            step *= 0.5
        trace.append(best)
        if step < step_min:
            break
    trace.extend([best] * (iters + 1 - len(trace)))
    return np.asarray(trace)


def _median_time(fn, reps):
    fn()                                        # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def fd_gradient(scene, subset, F, What, h=1e-6):
    """Full central-difference gradient at (F, {What}) in PyTorch's complex
    convention (dRe + j dIm), timed. Returns (seconds, [grad_F, grad_W...])."""
    leaves = [F] + list(What)
    grads = [torch.zeros_like(M) for M in leaves]
    t0 = time.perf_counter()
    with torch.no_grad():
        for li, M in enumerate(leaves):
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    for comp in (1.0 + 0.0j, 1.0j):
                        Mp = M.clone(); Mp[i, j] += h * comp
                        Mm = M.clone(); Mm[i, j] -= h * comp
                        Lp = [Mp if a == li else leaves[a] for a in range(len(leaves))]
                        Lm = [Mm if a == li else leaves[a] for a in range(len(leaves))]
                        Ip = float(_mi(scene, subset, Lp[0], Lp[1:]))
                        Im = float(_mi(scene, subset, Lm[0], Lm[1:]))
                        grads[li][i, j] += ((Ip - Im) / (2 * h)) * comp
    return time.perf_counter() - t0, grads


def ad_gradient(scene, subset, F, What):
    F_l = F.detach().requires_grad_(True)
    W_l = [w.detach().requires_grad_(True) for w in What]
    I = _mi(scene, subset, F_l, W_l)
    return torch.autograd.grad(I, [F_l] + W_l)


def main(selftest=False):
    plt = C._mpl()
    scene, names, coords = C.canonical_scene()
    n_tx = scene.n_ant("tx")
    n_rel = scene.n_ant(names[0])

    conv_iters = 40 if selftest else CONV_ITERS
    n_starts = 2 if selftest else N_STARTS
    time_ks = [1, 2] if selftest else TIME_KS
    reps = 2 if selftest else REPS

    # ---- (a) convergence & initialization on the greedy K=3 set -------------
    if selftest:
        idx = [11, 12, 13]
    else:
        idx = rgd.select_greedy(scene, names, K_CONV, iters=SEL_ITERS)
    subset = [names[i] for i in idx]
    print(f"convergence set (greedy K={K_CONV}): {idx}")
    traces = [pga_trace(scene, subset, iters=conv_iters, seed=r) for r in range(n_starts)]
    finals = np.array([t[-1] for t in traces])
    print(f"finals over {n_starts} random starts: mean={finals.mean():.2f} "
          f"std={finals.std():.2f}  min={finals.min():.2f}  best={finals.max():.2f} bits/s/Hz")

    # ---- (b) gradient efficiency vs #active relays --------------------------
    order = np.argsort(C.received_power_scores(scene, names))[::-1]
    t_fwd, t_ad, t_fd, n_par = [], [], [], []
    fd_err = None
    for k in time_ks:
        sub = [names[i] for i in order[:k]]
        F, What = _leaves(scene, sub, seed=0)
        n_par.append(2 * (F.numel() + sum(w.numel() for w in What)))

        def fwd():
            with torch.no_grad():
                _mi(scene, sub, F, What)
        t_fwd.append(_median_time(fwd, reps))
        t_ad.append(_median_time(lambda: ad_gradient(scene, sub, F, What), reps))
        tf, gfd = fd_gradient(scene, sub, F, What)
        t_fd.append(tf)
        if k == time_ks[0]:                     # AD-vs-FD exactness at the smallest K
            gad = ad_gradient(scene, sub, F, What)
            errs = [float((a - f).abs().max() / (a.abs().max() + 1e-30))
                    for a, f in zip(gad, gfd)]
            fd_err = max(errs)
        print(f"  K={k:2d}  params={n_par[-1]:4d}  forward={t_fwd[-1]*1e3:7.2f} ms  "
              f"AD sweep={t_ad[-1]*1e3:7.2f} ms ({t_ad[-1]/t_fwd[-1]:.1f}x fwd)  "
              f"FD={t_fd[-1]:7.2f} s ({t_fd[-1]/t_ad[-1]:.0f}x AD)")
    print(f"AD vs central-FD max relative gradient error (K={time_ks[0]}): {fd_err:.2e}")

    # ---- figure --------------------------------------------------------------
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.2, 4.3))
    it = np.arange(conv_iters + 1)
    for r, tr in enumerate(traces):
        axa.plot(it, tr, lw=1.0, color="tab:blue", alpha=0.45)
    axa.plot(it, np.maximum.reduce(traces), lw=2.0, color="tab:blue",
             label=f"best of {n_starts} random starts")
    axa.plot([], [], lw=1.0, color="tab:blue", alpha=0.45, label="single random starts")
    axa.set_xlabel("PGA iteration"); axa.set_ylabel("MI [bits/s/Hz]")
    axa.set_title(f"(a) convergence, greedy $K={K_CONV}$ set")
    axa.legend(loc="lower right", fontsize=9); axa.grid(alpha=0.3)
    axb.plot(time_ks, np.array(t_fd), "s-", color="tab:orange",
             label="central finite differences")
    axb.plot(time_ks, np.array(t_ad), "o-", color="tab:blue",
             label="one complex-AD sweep (all grads)")
    axb.plot(time_ks, np.array(t_fwd), "^--", color="0.5", label="one forward (MI)")
    for k, tf, npar in zip(time_ks, t_fd, n_par):
        axb.annotate(f"{npar}", (k, tf), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8, color="tab:orange")
    axb.set_yscale("log")
    axb.set_xlabel("active relays $K$"); axb.set_ylabel("wall-clock time [s]")
    axb.set_title("(b) gradient cost (labels: real-parameter count)")
    axb.legend(loc="center right", fontsize=9); axb.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = f"{C.OUT}/exp1_engine.pdf"; fig.savefig(p); print("saved", p)

    ok = (np.isfinite(finals).all() and finals.min() > 0
          and fd_err is not None and fd_err < 1e-4
          and all(a < f for a, f in zip(t_ad, t_fd)))
    return ok


if __name__ == "__main__":
    print("exp1_engine: convergence + gradient efficiency ...")
    raise SystemExit(0 if main(selftest="--selftest" in sys.argv) else 1)
