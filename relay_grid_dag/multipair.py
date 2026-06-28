"""Multi-pair NRG: relay selection under interference, via the cmi-dag CMI.

Several Tx/Rx pairs share the candidate relay grid. Each active relay forwards the
*superposition* of every transmitter (shared medium), and each receiver sees its
desired direct link, the cross (interference) direct links, and the relay forwards.
We assemble the multi-root linear-Gaussian DAG (one root per transmitter) from the
vendored near-field channels and read each pair's rate from the public ``cmi-dag``
conditional mutual information:

    R_u^TIN  = I(X_u; Y_u | {})              treat interference as noise (achievable)
    R_u^free = I(X_u; Y_u | X_{others})      interference-free upper bound (genie)

The gap ``R_u^free - R_u^TIN`` is the interference cost. Note the interference has
two paths -- the cross *direct* links AND the *shared relays* (an active relay
forwards every transmitter): a relay activated for one pair also carries another
pair's signal to its receiver.

Rates are in bits/s/Hz and are the idealized full-duplex value ``SE_FD`` (see the
duplexing convention in the README); a half-duplex realisation halves them.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from cmi_dag import compute_k_blocks_multiroot, conditional_mutual_information_from_k

from . import _nearfield as nfd
from .grid import N_RELAY, P_RELAY

__all__ = [
    "build_pair_scene", "pair_rates", "weighted_sum_rate", "min_rate",
    "select_greedy_sumrate", "select_exhaustive_sumrate", "received_power_pairs",
]

_LN2 = math.log(2.0)

# Defaults (overridable per call). Lengths in carrier wavelengths.
N_TX = N_RX = 4
P_TX = P_RELAY                       # equal Tx / per-relay power by default
SIGMA_R = SIGMA_D = 1.0
DIRECT_ATTEN = 0.3                   # desired direct link
CROSS_ATTEN = 0.3                    # interference (cross) direct link

# A default 2-pair interfering geometry: two pairs offset in y, candidate grid between.
DEFAULT_PAIRS = [((0.0, 2.0), (20.0, 2.0)), ((0.0, -2.0), (20.0, -2.0))]


def _eye(n):
    return torch.eye(n, dtype=nfd.DTYPE)


def grid_coords(nx=4, ny=4, xr=(6.0, 14.0), yr=(-4.0, 4.0)) -> np.ndarray:
    xs = np.linspace(xr[0], xr[1], nx)
    ys = np.linspace(yr[0], yr[1], ny)
    return np.asarray([(float(x), float(y)) for y in ys for x in xs], dtype=float)


def build_pair_scene(pairs=DEFAULT_PAIRS, *, coords: np.ndarray | None = None,
                     n_tx: int = N_TX, n_rx: int = N_RX, n_relay: int = N_RELAY,
                     model: str = "near"):
    """Scene with ``M`` Tx/Rx pairs (``tx{u}``/``rx{u}``) and an ``L``-candidate relay
    grid. ``pairs`` is a list of ``(tx_xy, rx_xy)``. Returns ``(scene, names, coords)``."""
    if coords is None:
        coords = grid_coords()
    s = nfd.Scene(model=model)
    for u, (tx_xy, rx_xy) in enumerate(pairs):
        s.add_node(f"tx{u}", list(tx_xy), n_tx, "source")
        s.add_node(f"rx{u}", list(rx_xy), n_rx, "sink")
    names = []
    for i, (x, y) in enumerate(coords):
        s.add_node(f"c{i}", [float(x), float(y)], n_relay, "relay")
        names.append(f"c{i}")
    return s, names, np.asarray(coords, dtype=float)


def pair_rates(scene, M: int, active, *, P_tx: float = P_TX, P_relay: float = P_RELAY,
               sigma_r: float = SIGMA_R, sigma_d: float = SIGMA_D,
               direct_atten: float = DIRECT_ATTEN, cross_atten: float = CROSS_ATTEN,
               n_tx: int = N_TX, n_rx: int = N_RX, jitter: float = 1e-9):
    """Per-pair TIN and interference-free rates (bits) for ``M`` pairs and an active
    relay-name list. Returns ``{"tin": [R_0..R_{M-1}], "free": [...]}``."""
    K = len(active)
    Xs = list(range(M))                              # roots X_0..X_{M-1}
    relays = list(range(M, M + K))                   # relay nodes
    Ys = list(range(M + K, 2 * M + K))               # sinks Y_0..Y_{M-1}
    Sx = [(P_tx / n_tx) * _eye(n_tx) for _ in range(M)]

    edge, parents, noise = {}, {}, {}
    gains = {}
    for r, nm in zip(relays, active):
        Hin = [scene.channel(f"tx{u}", nm) for u in range(M)]     # (n_relay, n_tx)
        Krr = sum((Hin[u] @ Sx[u] @ Hin[u].conj().T for u in range(M)),
                  sigma_r ** 2 * _eye(scene.n_ant(nm)))
        gains[nm] = math.sqrt(P_relay / float(Krr.diagonal().real.sum()))
        for u in range(M):
            edge[(r, Xs[u])] = Hin[u]
        parents[r] = list(Xs)
        noise[r] = sigma_r ** 2 * _eye(scene.n_ant(nm))

    for u, Yu in enumerate(Ys):
        parents[Yu] = list(Xs)
        for v in range(M):
            atten = direct_atten if v == u else cross_atten
            edge[(Yu, Xs[v])] = atten * scene.channel(f"tx{v}", f"rx{u}")
        for r, nm in zip(relays, active):
            edge[(Yu, r)] = gains[nm] * scene.channel(nm, f"rx{u}")
            parents[Yu].append(r)
        noise[Yu] = sigma_d ** 2 * _eye(n_rx)

    Kb = compute_k_blocks_multiroot(
        num_nodes=2 * M + K, roots=Xs, parents=parents, edge_mats=edge,
        root_covs={u: Sx[u] for u in range(M)}, noise_covs=noise)

    def cmi(A, B, C):
        return conditional_mutual_information_from_k(Kb, A, B, C, jitter=jitter).item() / _LN2

    tin = [cmi([Xs[u]], [Ys[u]], []) for u in range(M)]
    free = [cmi([Xs[u]], [Ys[u]], [Xs[v] for v in range(M) if v != u]) for u in range(M)]
    return {"tin": tin, "free": free}


# --------------------------------------------------------------------------- #
# Objectives and selection (sum-rate / max-min over the per-pair TIN rates).    #
# --------------------------------------------------------------------------- #
def weighted_sum_rate(scene, M, active, *, weights=None, **kw) -> float:
    """Weighted sum of per-pair TIN rates ``Σ_u w_u R_u`` (bits, FD)."""
    tin = pair_rates(scene, M, active, **kw)["tin"]
    w = [1.0] * M if weights is None else weights
    return float(sum(w[u] * tin[u] for u in range(M)))


def min_rate(scene, M, active, **kw) -> float:
    """Max-min objective: the smallest per-pair TIN rate ``min_u R_u`` (bits, FD)."""
    return float(min(pair_rates(scene, M, active, **kw)["tin"]))


def _greedy(score, L, K):
    chosen, remaining = [], set(range(L))
    for _ in range(K):
        best_j, best = None, -np.inf
        for j in remaining:
            v = score(chosen + [j])
            if v > best:
                best, best_j = v, j
        chosen.append(best_j); remaining.discard(best_j)
    return sorted(chosen)


def select_greedy_sumrate(scene, M, names, K, *, objective=weighted_sum_rate, **kw):
    """Forward greedy on the multi-pair objective (default: weighted sum-rate)."""
    score = lambda idxs: objective(scene, M, [names[i] for i in idxs], **kw)
    return _greedy(score, len(names), K)


def select_exhaustive_sumrate(scene, M, names, K, *, objective=weighted_sum_rate, **kw):
    """Brute-force the best K-subset for the multi-pair objective. Small L only."""
    best_set, best = None, -np.inf
    for combo in itertools.combinations(range(len(names)), K):
        v = objective(scene, M, [names[i] for i in combo], **kw)
        if v > best:
            best, best_set = v, combo
    return sorted(best_set), best


def received_power_pairs(scene, M, names, K) -> list[int]:
    """Interference-blind baseline: rank candidates by the desired two-hop cascade
    gain summed over pairs, take the top K (ignores the cross/interference paths)."""
    scores = np.zeros(len(names))
    with torch.no_grad():
        for i, nm in enumerate(names):
            for u in range(M):
                casc = scene.channel(nm, f"rx{u}") @ scene.channel(f"tx{u}", nm)
                scores[i] += float((casc.abs() ** 2).sum())
    return sorted(np.argsort(scores)[-K:].tolist())
