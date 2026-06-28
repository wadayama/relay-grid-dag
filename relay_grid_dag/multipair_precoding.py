"""Multi-pair precoded NRG: optimize the Tx precoders ``{F_u}`` and the relay
matrices ``{W_l}`` for a multi-pair objective via the ``cmi-dag`` conditional MI.

A set of Tx/Rx pairs shares the candidate relay grid. The DAG (roots ``X_u``, relay
nodes, sinks ``Y_u``) has, for active pairs ``{0,1}``:

    Tx_u -> Relays           (every active Tx reaches every active relay; superposition)
    Relays -> Rx_u           (every active relay forwards to every active Rx)
    Tx_v -> Rx_u             (all direct links: desired ``v=u`` and cross ``v!=u``)

Each pair's rate is the treat-interference-as-noise (TIN) MI ``R_u = I(X_u;Y_u)``
(interference from the other pairs -- via cross direct links and the shared relays --
folded into the noise). The objective is the TIN sum-rate ``Σ_u R_u`` or the max-min
``min_u R_u``. Matrix relays supply spatial degrees of freedom for interference
management that a scalar AF relay (``multipair.py``, the baseline) cannot.

The active pairs are given as ``pairs`` -- a list of pair indices (e.g. ``[0, 1]`` or
``[1]``) selecting nodes ``tx{u}``/``rx{u}``; an ``int`` ``M`` is shorthand for
``range(M)``. Same engine principle as ``precoding.py``: random-initialized
multi-start projected gradient ascent in the whitened relay parametrization
(``K_ll = sum_u (H_sr,u F_u)(.)^H + sigma_r^2 I`` couples all ``F_u``), extended to
the multi-root DAG. Non-convex: the result is a stationary point (an achievable
sum-rate), not a certified optimum. Rates are bits/s/Hz, idealized full-duplex.
"""
from __future__ import annotations

import math

import torch

from cmi_dag import compute_k_blocks_multiroot, conditional_mutual_information_from_k

from ._nearfield.channel import DTYPE
from .precoding import _proj_ball, _physical_W, _randc
from .multipair import P_TX, P_RELAY, SIGMA_R, SIGMA_D, DIRECT_ATTEN, CROSS_ATTEN

__all__ = ["optimize_multipair", "multipair_tin"]

_LN2 = math.log(2.0)


def _eye(n):
    return torch.eye(n, dtype=DTYPE)


def _as_pairs(pairs):
    """Normalize ``pairs`` (a list of pair indices, or an ``int`` M for range(M))."""
    return list(range(pairs)) if isinstance(pairs, int) else list(pairs)


def _relay_chols_mp(scene, pairs, active, F_list, sigma_r):
    """Cholesky ``L_l`` of each relay's received covariance
    ``K_ll = sum_k (H_sr,pairs[k] F_k)(.)^H + sigma_r^2 I`` (couples all ``F_k``)."""
    pairs = _as_pairs(pairs)
    Ls = []
    for nm in active:
        n = scene.n_ant(nm)
        K = (sigma_r ** 2) * _eye(n)
        for k, u in enumerate(pairs):
            A = scene.channel(f"tx{u}", nm) @ F_list[k]
            K = K + A @ A.mH
        Ls.append(torch.linalg.cholesky(K))
    return Ls


def multipair_tin(scene, pairs, active, F_list, Wphys, *, sigma_r=SIGMA_R, sigma_d=SIGMA_D,
                  direct_atten=DIRECT_ATTEN, cross_atten=CROSS_ATTEN):
    """Per-pair TIN MI ``[R_k for k in pairs]`` (bits, differentiable) for Tx precoders
    ``F_list`` and physical relay matrices ``Wphys`` on the multi-root DAG."""
    pairs = _as_pairs(pairs)
    P, K = len(pairs), len(active)
    Xs = list(range(P)); relays = list(range(P, P + K)); Ys = list(range(P + K, 2 * P + K))
    Sx = {k: _eye(scene.n_ant(f"tx{u}")) for k, u in enumerate(pairs)}   # input cov I
    edge, parents, noise = {}, {}, {}
    for r, nm in zip(relays, active):
        parents[r] = list(Xs)
        for k, u in enumerate(pairs):
            edge[(r, Xs[k])] = scene.channel(f"tx{u}", nm) @ F_list[k]   # precoded reception
        noise[r] = (sigma_r ** 2) * _eye(scene.n_ant(nm))
    for k, u in enumerate(pairs):
        Yu = Ys[k]
        parents[Yu] = list(Xs)
        for kv, v in enumerate(pairs):
            at = direct_atten if v == u else cross_atten
            edge[(Yu, Xs[kv])] = at * (scene.channel(f"tx{v}", f"rx{u}") @ F_list[kv])
        for j, (r, nm) in enumerate(zip(relays, active)):
            edge[(Yu, r)] = scene.channel(nm, f"rx{u}") @ Wphys[j]       # matrix relay forward
            parents[Yu].append(r)
        noise[Yu] = (sigma_d ** 2) * _eye(scene.n_ant(f"rx{u}"))
    Kb = compute_k_blocks_multiroot(num_nodes=2 * P + K, roots=Xs, parents=parents,
                                    edge_mats=edge, root_covs=Sx, noise_covs=noise)
    return [conditional_mutual_information_from_k(Kb, [Xs[k]], [Ys[k]], [], jitter=1e-9) / _LN2
            for k in range(P)]


def _objective(tin, obj):
    return torch.stack(tin).min() if obj == "minrate" else sum(tin)


def _eval(scene, pairs, active, F_list, What_list, p, obj):
    Ls = _relay_chols_mp(scene, pairs, active, F_list, p["sigma_r"])
    Wphys = [_physical_W(w, L) for w, L in zip(What_list, Ls)]
    tin = multipair_tin(scene, pairs, active, F_list, Wphys, sigma_r=p["sigma_r"],
                        sigma_d=p["sigma_d"], direct_atten=p["direct_atten"],
                        cross_atten=p["cross_atten"])
    return _objective(tin, obj), tin


def _build_init(scene, pairs, active, *, P_tx, P_relay, sigma_r, F_init, W_init, g):
    F0 = []
    for k, u in enumerate(pairs):
        n = scene.n_ant(f"tx{u}")
        if F_init is not None and F_init[k] is not None:
            F0.append(_proj_ball(F_init[k].detach().to(DTYPE).clone(), P_tx))
        else:
            F0.append(_proj_ball(_randc(n, n, g), P_tx))
    Ls = _relay_chols_mp(scene, pairs, active, F0, sigma_r)
    What0 = []
    for j, (nm, L) in enumerate(zip(active, Ls)):
        n = scene.n_ant(nm)
        if W_init is not None and j < len(W_init) and W_init[j] is not None:
            What0.append(_proj_ball(W_init[j].detach().to(DTYPE) @ L, P_relay))
        else:
            What0.append(_proj_ball(_randc(n, n, g), P_relay))
    return F0, What0


def _optimize_single(scene, pairs, active, F0, What0, *, p, obj, iters, step, step_min):
    P_tx, P_relay = p["P_tx"], p["P_relay"]
    F = [f.detach().clone() for f in F0]
    What = [w.detach().clone() for w in What0]

    def value(F_, What_):
        with torch.no_grad():
            return float(_eval(scene, pairs, active, F_, What_, p, obj)[0])

    def physical(F_, What_):
        Ls = _relay_chols_mp(scene, pairs, active, F_, p["sigma_r"])
        return [_physical_W(w, L).detach().clone() for w, L in zip(What_, Ls)]

    def tins(F_, What_):
        with torch.no_grad():
            return [float(t) for t in _eval(scene, pairs, active, F_, What_, p, obj)[1]]

    best = (value(F, What), [f.detach().clone() for f in F], physical(F, What), tins(F, What))
    for _ in range(iters):
        F_l = [f.detach().requires_grad_(True) for f in F]
        What_l = [w.detach().requires_grad_(True) for w in What]
        val, _ = _eval(scene, pairs, active, F_l, What_l, p, obj)
        cur = float(val.detach())
        grads = torch.autograd.grad(val, F_l + What_l)
        nF = len(F)
        gF, gW = grads[:nF], grads[nF:]
        with torch.no_grad():
            F_new = [_proj_ball(f + step * g, P_tx) for f, g in zip(F, gF)]
            What_new = [_proj_ball(w + step * g, P_relay) for w, g in zip(What, gW)]
        new = value(F_new, What_new)
        if new >= cur:
            F, What = F_new, What_new
            step *= 1.1
            if new > best[0]:
                best = (new, [f.detach().clone() for f in F], physical(F, What), tins(F, What))
        else:
            step *= 0.5
            if step < step_min:
                break
    return best


def optimize_multipair(scene, pairs, active, *, P_tx=P_TX, P_relay=P_RELAY,
                       sigma_r=SIGMA_R, sigma_d=SIGMA_D, direct_atten=DIRECT_ATTEN,
                       cross_atten=CROSS_ATTEN, objective="sumrate",
                       iters=300, step=0.05, step_min=1e-8, restarts=1, seed=0,
                       F_init=None, W_init=None):
    """Optimize ``{F_k}, {W_l}`` for the multi-pair TIN objective (``'sumrate'`` or
    ``'minrate'``) over the active ``pairs`` (a list of pair indices, or an int M for
    ``range(M)``) on the shared relay grid. Random-initialized multi-start PGA in the
    whitened relay parametrization. Returns
    ``(best_obj_bits, F_list, Wphys_list, per_pair_tin_bits)`` with ``F_list`` /
    ``per_pair_tin_bits`` aligned to ``pairs``.
    """
    pairs = _as_pairs(pairs)
    active = list(active)
    p = dict(P_tx=P_tx, P_relay=P_relay, sigma_r=sigma_r, sigma_d=sigma_d,
             direct_atten=direct_atten, cross_atten=cross_atten)
    best = None
    for r in range(max(1, restarts)):
        g = torch.Generator().manual_seed(seed + r)
        F0, What0 = _build_init(scene, pairs, active, P_tx=P_tx, P_relay=P_relay, sigma_r=sigma_r,
                                F_init=(F_init if r == 0 else None),
                                W_init=(W_init if r == 0 else None), g=g)
        cand = _optimize_single(scene, pairs, active, F0, What0, p=p, obj=objective,
                                iters=iters, step=step, step_min=step_min)
        if best is None or cand[0] > best[0]:
            best = cand
    return best
