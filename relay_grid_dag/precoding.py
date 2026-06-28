"""Differentiable joint Tx/relay precoding --- the inner engine (SPEC.md Sec. 4).

Maximizes the merge-channel mutual information over the Tx precoder ``F`` and the
per-relay matrices ``{W_l}`` by projected gradient ascent in the *whitened* relay
parametrization ``What_l = W_l L_l`` (with ``K_ll = L_l L_l^H``), so each power
budget is an exact Frobenius ball and no augmented Lagrangian is needed. The
physical relay matrices ``W_l = What_l L_l^{-1}`` are recovered *inside* the
autograd graph, so the ``F``-dependence of ``L_l`` flows through the same backward
sweep (SPEC.md Remark 1 / Sec. 4). Random-initialized (seeded; multi-start via
``restarts``, the target of the future GPU pga-toolbox). The conventional scalar-AF
relay (``subset_mi``) is the engine-independent baseline for the precoding gain.
"""
from __future__ import annotations

import math

import torch

from ._nearfield.channel import DTYPE, K_WAVE
from ._nearfield.dag import mi, multirelay_merge


def _relay_chols(scene, src, subset, F, sigma_r, k_wave):
    """Cholesky factors ``L_l`` of each relay's received covariance ``K_ll(F)``.

    With a precoder ``F`` the source input covariance is ``I``, so the relay
    reception covariance is ``K_ll = (H_sr F)(H_sr F)^H + sigma_r^2 I`` (HPD).
    """
    Ls = []
    for nm in subset:
        A = scene.channel(src, nm, k_wave=k_wave) @ F
        n = A.shape[0]
        K = A @ A.mH + (sigma_r ** 2) * torch.eye(n, dtype=DTYPE)
        Ls.append(torch.linalg.cholesky(K))
    return Ls


def _physical_W(What, L):
    """Physical relay matrix ``W = What L^{-1}`` via a triangular solve.

    Solves ``W^H = (L^H)^{-1} What^H`` (``L^H`` upper-triangular), no explicit
    inverse.
    """
    return torch.linalg.solve_triangular(L.mH, What.mH, upper=True).mH


def _merge_mi(scene, src, subset, dst, F, What, *, sigma_r, sigma_d, k_wave):
    """Merge-channel MI for the whitened leaves ``(F, {What_l})``.

    Differentiable in ``F`` (through the channels, the whitening ``L_l``, and the
    de-whitening) and in every ``What_l``.
    """
    Ls = _relay_chols(scene, src, subset, F, sigma_r, k_wave)
    Ws = [_physical_W(w, L) for w, L in zip(What, Ls)]
    spec = multirelay_merge(scene, src, subset, dst, sigma_r=sigma_r, sigma_d=sigma_d,
                            k_wave=k_wave, precoder=F,
                            relay_mats=Ws if subset else None)
    return mi(spec)


def _proj_ball(M, P):
    """Euclidean projection onto the Frobenius ball ``{‖M‖_F^2 <= P}`` (a scaling)."""
    nrm2 = float((M.abs() ** 2).sum())
    if nrm2 <= P:
        return M
    return M * math.sqrt(P / nrm2)


def _randc(a, b, g):
    """Random complex matrix ~ CN(0, I) via generator ``g`` (reproducible)."""
    return torch.complex(torch.randn(a, b, generator=g, dtype=torch.float64),
                         torch.randn(a, b, generator=g, dtype=torch.float64))


def _build_init(scene, src, subset, *, P_tx, P_relay, sigma_r, k_wave, d, F_init, W_init, g):
    """Initial ``(F, {What})`` for one start. RANDOM by default (generator ``g``);
    a supplied warm start ``F_init`` / ``W_init`` (physical) overrides where given
    --- a zero ``W_init[i]`` leaves relay ``i`` off. Each block is projected onto
    its power ball.
    """
    n_tx = scene.n_ant(src)
    if F_init is not None:
        F0 = _proj_ball(F_init.detach().to(DTYPE).clone(), P_tx)
    else:
        F0 = _proj_ball(_randc(n_tx, d, g), P_tx)
    Ls = _relay_chols(scene, src, subset, F0, sigma_r, k_wave)
    What0 = []
    for i, L in enumerate(Ls):
        if W_init is not None and i < len(W_init) and W_init[i] is not None:
            What0.append(_proj_ball(W_init[i].detach().to(DTYPE) @ L, P_relay))
        else:
            n = L.shape[0]
            What0.append(_proj_ball(_randc(n, n, g), P_relay))
    return F0, What0


def _optimize_single(scene, src, subset, dst, F0, What0, *, P_tx, P_relay,
                     sigma_r, sigma_d, k_wave, iters, step, step_min, freeze_F=False):
    """One PGA run from the given start ``(F0, {What0})``; adaptive step
    (grow on accept, halve on reject) with best-iterate tracking. Returns
    ``(best_mi_bits, F, W_list)``. With ``freeze_F`` the Tx precoder is held at
    ``F0`` (only the relay matrices are optimized --- the relay-only ablation)."""
    F = F0.detach().clone()
    What = [w.detach().clone() for w in What0]

    def value(F_, What_):
        with torch.no_grad():
            return float(_merge_mi(scene, src, subset, dst, F_, What_,
                                   sigma_r=sigma_r, sigma_d=sigma_d, k_wave=k_wave))

    def physical(F_, What_):
        Ls = _relay_chols(scene, src, subset, F_, sigma_r, k_wave)
        return [_physical_W(w, L).detach().clone() for w, L in zip(What_, Ls)]

    best = (value(F, What), F.detach().clone(), physical(F, What))
    for _ in range(iters):
        F_l = F.detach().requires_grad_(True)
        What_l = [w.detach().requires_grad_(True) for w in What]
        I = _merge_mi(scene, src, subset, dst, F_l, What_l,
                      sigma_r=sigma_r, sigma_d=sigma_d, k_wave=k_wave)
        I_cur = float(I.detach())
        grads = torch.autograd.grad(I, [F_l] + What_l)
        with torch.no_grad():
            F_new = F if freeze_F else _proj_ball(F + step * grads[0], P_tx)
            What_new = [_proj_ball(w + step * gg, P_relay)
                        for w, gg in zip(What, grads[1:])]
        I_new = value(F_new, What_new)
        if I_new >= I_cur:
            F, What = F_new, What_new
            step *= 1.1
            if I_new > best[0]:
                best = (I_new, F.detach().clone(), physical(F, What))
        else:
            step *= 0.5
            if step < step_min:
                break
    return best


def optimize_precoders(scene, src, subset, dst, *, P_tx=100.0, P_relay=100.0,
                       sigma_r=1.0, sigma_d=1.0, k_wave=K_WAVE, d=None,
                       iters=400, step=0.05, step_min=1e-8,
                       restarts=1, seed=0, F_init=None, W_init=None, freeze_F=False):
    """Maximize the merge-channel MI over ``(F, {W_l})`` --- the SPEC.md Sec. 4 engine.

    Initialization is RANDOM (seeded, reproducible) by default; ``restarts`` runs
    that many independent random starts and keeps the best (multi-start --- the
    starts are independent and embarrassingly parallel, the target of the future
    GPU ``pga-toolbox``). The step is adaptive and the best iterate is tracked.

    ``F_init`` / ``W_init`` (physical) override the first start's init for warm
    starting --- e.g. greedy selection threads a smaller active set's optimum, and
    a zero ``W_init`` entry leaves a relay off. Returns ``(best_mi_bits, F, W_list)``.

    Because the start is random, the returned MI is the best stationary point
    found, not a certified optimum; it exceeds the conventional scalar-AF relay
    empirically (large margin in practice), not by construction.
    """
    subset = list(subset)
    n_tx = scene.n_ant(src)
    d = n_tx if d is None else d
    best = None
    for r in range(max(1, restarts)):
        g = torch.Generator().manual_seed(seed + r)
        F0, What0 = _build_init(scene, src, subset, P_tx=P_tx, P_relay=P_relay,
                                sigma_r=sigma_r, k_wave=k_wave, d=d,
                                F_init=(F_init if (r == 0 or freeze_F) else None),
                                W_init=(W_init if r == 0 else None), g=g)
        cand = _optimize_single(scene, src, subset, dst, F0, What0, P_tx=P_tx,
                                P_relay=P_relay, sigma_r=sigma_r, sigma_d=sigma_d,
                                k_wave=k_wave, iters=iters, step=step, step_min=step_min,
                                freeze_F=freeze_F)
        if best is None or cand[0] > best[0]:
            best = cand
    return best


def optimized_mi(scene, src, subset, dst, **kw):
    """The optimized-MI objective ``f(S)`` in bits --- the merge-channel MI
    maximized over the Tx and relay precoders. Convenience wrapper returning only
    the scalar; see :func:`optimize_precoders` for the precoders too."""
    return optimize_precoders(scene, src, subset, dst, **kw)[0]
