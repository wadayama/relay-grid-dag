"""Relay-selection layer: activate K of L candidate relays to maximise the spectral efficiency.

One model, one selection family. The objective is the optimized MI
``f(S)`` --- the merge-channel MI maximized over the Tx and relay precoders
(:func:`relay_grid_dag.precoding.optimize_precoders`, random-initialized
multi-start); the selectors here choose *which* relays to activate. The
conventional scalar-AF relay (``subset_mi``) is the engine-independent baseline
against which the precoding gain is measured.

Strategies: scalar-AF baseline MI (``subset_mi``), received-power / distance
baselines, forward greedy (warm-started), exhaustive oracle, 1-swap local search,
and the continuous-position PGA used by continuous-to-discrete. All MI
values are in bits/s/Hz.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from . import _nearfield as nfd
from ._nearfield.channel import K_WAVE
from .grid import N_RELAY, P_RELAY, TX_XY, RX_XY, N_TX, N_RX, DIRECT_ATTEN
from .precoding import optimize_precoders

__all__ = [
    "subset_mi", "direct_only_mi", "received_power_scores",
    "select_received_power", "select_distance",
    "select_greedy", "select_exhaustive", "swap_search",
    "random_subset_stats", "continuous_relays", "round_to_grid",
]


# --------------------------------------------------------------------------- #
# Scalar-AF baseline MI (the conventional relay; computed directly, the bound).  #
# --------------------------------------------------------------------------- #
def subset_mi(scene, subset, *, src: str = "tx", dst: str = "rx",
              k_wave: float = K_WAVE, P_tx: float = 100.0,
              P_relay: float = P_RELAY) -> float:
    """Scalar-AF baseline MI in bits for the diamond formed by activating
    ``subset`` relays plus the (attenuated) direct path. This is the conventional
    amplify-and-forward relay (a power-normalized scalar gain, no precoder
    optimization). ``subset=[]`` is the direct link."""
    subset = list(subset)
    if len(set(subset)) != len(subset):
        raise ValueError(f"duplicate relay names in subset: {subset}")
    with torch.no_grad():
        spec = nfd.multirelay_merge(scene, src, subset, dst,
                                    P_tx=P_tx, P_relay=P_relay, k_wave=k_wave)
        return nfd.mi(spec).item()


def direct_only_mi(scene, *, src: str = "tx", dst: str = "rx",
                   k_wave: float = K_WAVE, P_tx: float = 100.0,
                   P_relay: float = P_RELAY) -> float:
    return subset_mi(scene, [], src=src, dst=dst, k_wave=k_wave,
                     P_tx=P_tx, P_relay=P_relay)


# --------------------------------------------------------------------------- #
# Naive baselines (model-agnostic; no inner optimization during selection).    #
# --------------------------------------------------------------------------- #
def received_power_scores(scene, names) -> np.ndarray:
    """Per-candidate two-hop cascade gain ``||H_rd @ H_tr||_F^2`` -- the naive
    received-power proxy."""
    scores = np.zeros(len(names))
    with torch.no_grad():
        for i, nm in enumerate(names):
            h_tr = scene.channel("tx", nm)
            h_rd = scene.channel(nm, "rx")
            scores[i] = float(((h_rd @ h_tr).abs() ** 2).sum())
    return scores


def select_received_power(scene, names, K: int) -> list[int]:
    """Indices of the top-K candidates by received-power score."""
    if K <= 0:
        return []
    return sorted(np.argsort(received_power_scores(scene, names))[-K:].tolist())


def select_distance(coords, K: int, tx=TX_XY, rx=RX_XY) -> list[int]:
    """Indices of the K candidates with the shortest two-hop path length
    ``||tx - g|| + ||g - rx||`` -- the naive distance-based baseline."""
    if K <= 0:
        return []
    tx = np.asarray(tx, float); rx = np.asarray(rx, float)
    d = np.linalg.norm(coords - tx, axis=1) + np.linalg.norm(coords - rx, axis=1)
    return sorted(np.argsort(d)[:K].tolist())


# --------------------------------------------------------------------------- #
# The selection family on the optimized MI f(S) (one model; iters knob).  #
# --------------------------------------------------------------------------- #
def select_greedy(scene, names, K: int, *, warm_start: bool = True, **engine_kw) -> list[int]:
    """Forward greedy on the optimized MI ``f(S)``: repeatedly add the
    candidate with the largest MI gain. ``O(K*L)`` inner optimizations.

    ``engine_kw`` is forwarded to :func:`optimize_precoders` (``iters``, ``P_tx``,
    ``P_relay``, ``k_wave``, ``restarts``, ...). With ``warm_start`` (default), each
    candidate is seeded from the current active set's optimum (the new relay random)."""
    if K > len(names):
        raise ValueError(f"K={K} exceeds the L={len(names)} candidates")
    chosen: list[int] = []
    remaining = set(range(len(names)))
    F_star, W_star = None, []
    for _ in range(K):
        best = None  # (cap, j, F, Ws)
        for j in remaining:
            subset = [names[i] for i in chosen + [j]]
            if warm_start and F_star is not None:
                cap, F, Ws = optimize_precoders(scene, "tx", subset, "rx",
                                                F_init=F_star, W_init=W_star + [None],
                                                **engine_kw)
            else:
                cap, F, Ws = optimize_precoders(scene, "tx", subset, "rx", **engine_kw)
            if best is None or cap > best[0]:
                best = (cap, j, F, Ws)
        chosen.append(best[1])
        remaining.discard(best[1])
        F_star, W_star = best[2], best[3]
    return sorted(chosen)


def select_exhaustive(scene, names, K: int, **engine_kw):
    """Brute-force the best K-subset on the optimized MI. Returns
    ``(best_indices, best_mi)``. Feasible only at small scale:
    ``C(L,K)`` inner optimizations."""
    if K > len(names):
        raise ValueError(f"K={K} exceeds the L={len(names)} candidates")
    best = None  # (combo, cap)
    for combo in itertools.combinations(range(len(names)), K):
        cap = optimize_precoders(scene, "tx", [names[i] for i in combo], "rx",
                                 **engine_kw)[0]
        if best is None or cap > best[1]:
            best = (combo, cap)
    return sorted(best[0]), best[1]


def swap_search(scene, names, init, K: int, *, max_passes: int = 8, **engine_kw):
    """1-swap local search on the optimized MI from ``init`` (list of
    indices). Returns ``(improved_indices, mi)``; accepts any single swap
    that raises ``f(S)``. Used to polish a greedy or rounded start."""
    cur = list(init)
    cur_cap = optimize_precoders(scene, "tx", [names[i] for i in cur], "rx",
                                 **engine_kw)[0]
    for _ in range(max_passes):
        improved = False
        for a in list(cur):
            for m in range(len(names)):
                if m in cur:
                    continue
                trial = sorted([m if x == a else x for x in cur])
                cap = optimize_precoders(scene, "tx", [names[i] for i in trial], "rx",
                                         **engine_kw)[0]
                if cap > cur_cap + 1e-9:
                    cur, cur_cap, improved = trial, cap, True
        if not improved:
            break
    return sorted(cur), cur_cap


def random_subset_stats(scene, names, K: int, *, trials: int = 200,
                        seed: int = 0, k_wave: float = K_WAVE,
                        P_relay: float = P_RELAY):
    """Mean / std / max / min scalar-AF MI over ``trials`` random K-subsets
    (a baseline reference distribution; uses ``subset_mi``)."""
    rng = np.random.default_rng(seed)
    L = len(names)
    vals = np.array([
        subset_mi(scene, [names[i] for i in rng.choice(L, size=K, replace=False)],
                  k_wave=k_wave, P_relay=P_relay) for _ in range(trials)])
    return dict(mean=float(vals.mean()), std=float(vals.std()),
                max=float(vals.max()), min=float(vals.min()))


# --------------------------------------------------------------------------- #
# Continuous-to-discrete: position-gradient PGA then grid rounding.            #
# (Optimizes relay positions at the scalar-AF operating point; joint position+  #
#  precoder optimization is a brush-up-phase refinement.)                       #
# --------------------------------------------------------------------------- #
def continuous_relays(K: int, lo, hi, *, starts: int = 4, iters: int = 300,
                      step: float = 0.04, d_min: float = 1.8, model: str = "near",
                      direct_atten: float = DIRECT_ATTEN, P_relay: float = P_RELAY,
                      tx_xy=TX_XY, rx_xy=RX_XY, n_tx: int = N_TX, n_rx: int = N_RX,
                      n_relay: int = N_RELAY, seed0: int = 0, return_all: bool = False):
    """Multi-start position-gradient PGA on K virtual (off-grid) relay centres.

    Returns ``(final_centers (K,2), R_cont)`` for the best start, or -- with
    ``return_all=True`` -- a list of ``(final, R)`` per start sorted by R desc."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    best = None
    all_starts = []
    for st in range(starts):
        rng = np.random.default_rng(seed0 + st)
        s = nfd.Scene(model=model, direct_atten=direct_atten)
        s.add_node("tx", list(tx_xy), n_tx, "source")
        s.add_node("rx", list(rx_xy), n_rx, "sink")
        vn = []
        for i in range(K):
            start = lo + rng.random(2) * (hi - lo)
            s.add_node(f"v{i}", start.tolist(), n_relay, "relay", movable_node=True)
            vn.append(f"v{i}")
        for _ in range(iters):
            for c in s.params:
                if c.grad is not None:
                    c.grad.zero_()
            I = nfd.mi(nfd.multirelay_merge(s, "tx", vn, "rx", P_relay=P_relay))
            I.backward()
            with torch.no_grad():
                stepped = [c.detach() + step * c.grad for c in s.params]
                boxed = [nfd.project_box(c, lo.tolist(), hi.tolist()) for c in stepped]
                sep = nfd.project_min_separation(boxed, d_min)
            for v, c in zip(vn, sep):
                s.move(v, c.tolist())
        with torch.no_grad():
            R = nfd.mi(nfd.multirelay_merge(s, "tx", vn, "rx", P_relay=P_relay)).item()
        final = np.array([c.detach().tolist() for c in s.params])
        all_starts.append((final, R))
        if best is None or R > best[1]:
            best = (final, R)
    if return_all:
        return sorted(all_starts, key=lambda fr: fr[1], reverse=True)
    return best


def round_to_grid(final, coords) -> list[int]:
    """Nearest candidate per virtual relay, resolving collisions to the next
    nearest free grid point."""
    idx: list[int] = []
    for p in final:
        for cand in np.argsort(((coords - p) ** 2).sum(1)):
            if int(cand) not in idx:
                idx.append(int(cand))
                break
    return sorted(idx)
