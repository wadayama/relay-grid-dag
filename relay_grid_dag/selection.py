"""Relay-selection layer: activate K of L candidate relays to maximise MI.

These are plain combinatorics on top of ``mi(multirelay_merge(...))`` from the
vendored near-field engine -- the part that distinguishes NF-PRG from a movable
relay: a fixed candidate grid, from which a few sites are activated. Strategies:
received-power / distance baselines, forward greedy, exhaustive oracle, 1-swap
local search, and the continuous-position PGA used by continuous-to-discrete.

All MI values are in bits/s/Hz (the engine's ``mi`` returns bits by default).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from . import _nearfield as nfd
from ._nearfield.channel import K_WAVE
from .grid import N_RELAY, P_RELAY, TX_XY, RX_XY, N_TX, N_RX

__all__ = [
    "subset_mi", "direct_only_mi", "received_power_scores",
    "select_received_power", "select_distance", "select_greedy_mi",
    "select_exhaustive", "swap_search", "random_subset_stats",
    "continuous_relays", "round_to_grid",
]


# --------------------------------------------------------------------------- #
# MI of a chosen subset (the single physics call the rest is built on).        #
# --------------------------------------------------------------------------- #
def subset_mi(scene, subset, *, k_wave: float = K_WAVE, P_relay: float = P_RELAY) -> float:
    """I(X;Y) in bits for the diamond formed by activating ``subset`` relays plus
    the (attenuated) direct path. ``subset=[]`` is the direct-only link."""
    with torch.no_grad():
        spec = nfd.multirelay_merge(scene, "tx", list(subset), "rx",
                                    P_relay=P_relay, k_wave=k_wave)
        return nfd.mi(spec).item()


def direct_only_mi(scene, *, k_wave: float = K_WAVE) -> float:
    return subset_mi(scene, [], k_wave=k_wave)


# --------------------------------------------------------------------------- #
# Selection strategies.                                                        #
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


def select_greedy_mi(scene, names, K: int, *, k_wave: float = K_WAVE,
                     P_relay: float = P_RELAY) -> list[int]:
    """Forward greedy: repeatedly add the candidate with the largest MI gain.
    O(K*L) MI evaluations; deterministic; near-oracle in practice."""
    chosen: list[int] = []
    remaining = set(range(len(names)))
    for _ in range(K):
        best_j, best_mi = None, -np.inf
        for j in remaining:
            mi = subset_mi(scene, [names[i] for i in chosen + [j]], k_wave=k_wave,
                           P_relay=P_relay)
            if mi > best_mi:
                best_mi, best_j = mi, j
        chosen.append(best_j)
        remaining.discard(best_j)
    return sorted(chosen)


def select_exhaustive(scene, names, K: int, *, k_wave: float = K_WAVE):
    """Brute-force the best K-subset. Returns ``(best_indices, best_mi)``.
    Feasible only at small scale: C(L,K) MI evaluations."""
    best_set, best_mi = None, -np.inf
    for combo in itertools.combinations(range(len(names)), K):
        mi = subset_mi(scene, [names[i] for i in combo], k_wave=k_wave)
        if mi > best_mi:
            best_mi, best_set = mi, combo
    return sorted(best_set), best_mi


def swap_search(scene, names, init, K: int, *, k_wave: float = K_WAVE,
                max_passes: int = 8):
    """1-swap local search from ``init`` (list of indices). Returns
    ``(improved_indices, mi)``; greedily accepts any single swap that raises MI."""
    cur = list(init)
    cur_mi = subset_mi(scene, [names[i] for i in cur], k_wave=k_wave)
    for _ in range(max_passes):
        improved = False
        for a in list(cur):
            for m in range(len(names)):
                if m in cur:
                    continue
                trial = sorted([m if x == a else x for x in cur])
                mi = subset_mi(scene, [names[i] for i in trial], k_wave=k_wave)
                if mi > cur_mi + 1e-9:
                    cur, cur_mi, improved = trial, mi, True
        if not improved:
            break
    return sorted(cur), cur_mi


def random_subset_stats(scene, names, K: int, *, trials: int = 200,
                        seed: int = 0, k_wave: float = K_WAVE):
    """Mean / std / max / min MI over ``trials`` random K-subsets."""
    rng = np.random.default_rng(seed)
    L = len(names)
    vals = np.array([
        subset_mi(scene, [names[i] for i in rng.choice(L, size=K, replace=False)],
                  k_wave=k_wave) for _ in range(trials)])
    return dict(mean=float(vals.mean()), std=float(vals.std()),
                max=float(vals.max()), min=float(vals.min()))


# --------------------------------------------------------------------------- #
# Continuous-to-discrete: position-gradient PGA then grid rounding.            #
# --------------------------------------------------------------------------- #
def continuous_relays(K: int, lo, hi, *, starts: int = 4, iters: int = 300,
                      step: float = 0.04, d_min: float = 1.8, model: str = "near",
                      direct_atten: float = 0.3, P_relay: float = P_RELAY,
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
        final = np.array([s._nodes[v]["center"].detach().tolist() for v in vn])
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
