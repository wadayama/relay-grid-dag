"""Linear-Gaussian DAG-spec emitters and a thin mutual-information wrapper.

Each emitter turns a :class:`~relay_grid_dag._nearfield.scene.Scene` (plus power/noise
settings) into a tuple

    spec = (num_nodes, parents, edge_mats, input_cov, noise_covs)

ready to splat into ``gaussian_dag.compute_k_blocks``. This is the decoupling
boundary: nearfield-dag builds the channels and the DAG structure; gaussian-dag
runs the K-recursion and the log-det mutual information. The K-recursion is never
reimplemented here.

Conventions inherited from gaussian-dag: node 0 is the unique root ``X``; an edge
``i -> j`` (with ``i < j``) is keyed ``edge_mats[(j, i)] = A_{ji}`` of shape
``(d_j, d_i)``; ``parents[j]`` lists parent indices.
"""
from __future__ import annotations

import math

import torch

from gaussian_dag import compute_k_blocks, mutual_information_from_k

from .channel import DTYPE, K_WAVE
from .relay import af_gain

DagSpec = tuple[int, dict, dict, torch.Tensor, dict]


def _eye(n: int) -> torch.Tensor:
    return torch.eye(n, dtype=DTYPE)


def _source_cov(scene, src, P_tx, precoder):
    """Input covariance + the precoder applied on each outgoing edge.

    Without a precoder the source emits the antenna signal directly with
    isotropic covariance ``(P_tx/n_tx) I``; with a precoder ``F`` the source is
    ``s ~ CN(0, I)`` and ``F`` multiplies every channel leaving the source.
    Returns ``(input_cov, apply)`` where ``apply(H) = H`` or ``H @ F``.
    """
    if precoder is None:
        n_tx = scene.n_ant(src)
        return (P_tx / n_tx) * _eye(n_tx), (lambda H: H)
    n_s = precoder.shape[1]
    return _eye(n_s), (lambda H: H @ precoder)


def single_link(scene, src, dst, *, P_tx=100.0, sigma=1.0, precoder=None,
                k_wave=K_WAVE) -> DagSpec:
    """2-node link ``X -> Y`` (single near/far-field channel).

    ``k_wave`` selects the subcarrier (default carrier / single-carrier).
    """
    input_cov, apply = _source_cov(scene, src, P_tx, precoder)
    H = apply(scene.channel(src, dst, k_wave=k_wave))
    edge_mats = {(1, 0): H}
    noise = {1: (sigma ** 2) * _eye(scene.n_ant(dst))}
    return 2, {1: [0]}, edge_mats, input_cov, noise


def diamond(scene, src, relay, dst, *, P_tx=100.0, P_relay=100.0,
            sigma_r=1.0, sigma_d=1.0, direct=True, precoder=None,
            relay_gain=None, k_wave=K_WAVE) -> DagSpec:
    """3-node diamond ``X, relay, Y``: direct + amplify-and-forward relayed path.

    Node 0 = X, 1 = relay (received), 2 = Y (merges direct and relayed). The AF
    gain is power-normalised to ``P_relay`` given the actual first-hop edge,
    unless ``relay_gain`` is supplied (a fixed scalar gain) -- a frozen gain makes
    the diamond linear in ``precoder``, so water-filling is the exact optimum.
    ``k_wave`` selects the subcarrier (default carrier / single-carrier).
    """
    input_cov, apply = _source_cov(scene, src, P_tx, precoder)
    A10 = apply(scene.channel(src, relay, k_wave=k_wave))
    H_rd = scene.channel(relay, dst, k_wave=k_wave)
    g = af_gain(A10, input_cov, sigma_r, P_relay) if relay_gain is None else relay_gain
    edge_mats = {(1, 0): A10, (2, 1): g * H_rd}
    parents = {1: [0], 2: [1]}
    if direct:
        edge_mats[(2, 0)] = apply(scene.channel(src, dst, atten=scene.direct_atten,
                                                k_wave=k_wave))
        parents[2] = [0, 1]
    noise = {1: (sigma_r ** 2) * _eye(scene.n_ant(relay)),
             2: (sigma_d ** 2) * _eye(scene.n_ant(dst))}
    return 3, parents, edge_mats, input_cov, noise


def multirelay_merge(scene, src, relays, dst, *, P_tx=100.0, P_relay=100.0,
                     sigma_r=1.0, sigma_d=1.0, direct=True, k_wave=K_WAVE,
                     precoder=None, relay_mats=None) -> DagSpec:
    """Multi-branch merge: ``K`` AF relays + optional direct path.

    Nodes: 0 = X, 1..K = relays, K+1 = Y. The K-recursion carries the relay-relay
    cross-covariances at the merge node (shared source). ``k_wave`` selects the
    subcarrier (default carrier / single-carrier).

    Two optional precoding inputs turn the isotropic, scalar-AF model into the
    precoded merge channel of ``SPEC.md`` (the defaults reproduce the former
    exactly):

    - ``precoder``: a Tx precoder ``F`` (shape ``(n_tx, d)``). With ``F`` the root
      is ``X ~ CN(0, I_d)`` and ``F`` multiplies every source edge
      (``input_cov = I``, edge ``= H @ F``); without it the source emits with the
      isotropic ``input_cov = (P_tx/n_tx) I`` and unscaled edges.
    - ``relay_mats``: per-relay matrices ``[W_1, ..., W_K]`` (each ``(n_l, n_l)``),
      so the relay-to-destination edge is ``H_rd @ W`` instead of the scalar AF
      gain ``g * H_rd``. ``W = g I`` recovers the scalar case.

    The relay-power budget ``tr(W K_ll W^H) <= P_relay`` (with the relay received
    covariance ``K_ll = A_sr input_cov A_sr^H + sigma_r^2 I``) is enforced by the
    caller; the scalar default sets ``g = sqrt(P_relay / tr K_ll)`` and saturates it.
    """
    Kr = len(relays)
    input_cov, apply = _source_cov(scene, src, P_tx, precoder)
    Y = Kr + 1
    edge_mats: dict = {}
    parents: dict = {Y: []}
    noise: dict = {Y: (sigma_d ** 2) * _eye(scene.n_ant(dst))}
    if direct:
        edge_mats[(Y, 0)] = apply(scene.channel(src, dst, atten=scene.direct_atten,
                                                k_wave=k_wave))
        parents[Y].append(0)
    for i, relay in enumerate(relays, start=1):
        A_sr = apply(scene.channel(src, relay, k_wave=k_wave))   # precoded source->relay edge
        H_rd = scene.channel(relay, dst, k_wave=k_wave)
        if relay_mats is None:
            g = af_gain(A_sr, input_cov, sigma_r, P_relay)       # scalar AF gain
            edge_rd = g * H_rd
        else:
            edge_rd = H_rd @ relay_mats[i - 1].to(DTYPE)         # matrix relay precoder
        edge_mats[(i, 0)] = A_sr
        edge_mats[(Y, i)] = edge_rd
        parents[i] = [0]
        parents[Y].append(i)
        noise[i] = (sigma_r ** 2) * _eye(scene.n_ant(relay))
    return Y + 1, parents, edge_mats, input_cov, noise


def ris_branch(scene, src, ris, dst, *, P_tx=100.0, sigma_d=1.0, phi=None,
               direct=True, k_wave=K_WAVE) -> DagSpec:
    """2-node link with a passive RIS folded into the effective channel.

    ``H_eff = H_sd + H_rd diag(phi) H_sr`` (``phi`` defaults to all-ones = a
    transparent RIS). The RIS is passive and noiseless, so this stays a 2-node
    ``X -> Y`` spec. ``phi`` is a complex vector of length ``n_ris`` (unit-modulus
    in practice; not enforced here). ``k_wave`` selects the subcarrier (default
    carrier / single-carrier).
    """
    n_tx = scene.n_ant(src)
    input_cov = (P_tx / n_tx) * _eye(n_tx)
    H_sr = scene.channel(src, ris, k_wave=k_wave)
    H_rd = scene.channel(ris, dst, k_wave=k_wave)
    Phi = torch.diag(phi.to(DTYPE)) if phi is not None else _eye(scene.n_ant(ris))
    H_eff = H_rd @ Phi @ H_sr
    if direct:
        H_eff = H_eff + scene.channel(src, dst, atten=scene.direct_atten,
                                      k_wave=k_wave)
    edge_mats = {(1, 0): H_eff}
    noise = {1: (sigma_d ** 2) * _eye(scene.n_ant(dst))}
    return 2, {1: [0]}, edge_mats, input_cov, noise


def mi(spec: DagSpec, *, output_node=None, input_node=0, jitter=1e-9,
       bits=True) -> torch.Tensor:
    """Mutual information of a DAG spec via the gaussian-dag K-recursion.

    Returns a differentiable scalar in bits (or nats if ``bits=False``).
    ``output_node`` defaults to the last node ``num_nodes - 1``.
    """
    num_nodes = spec[0]
    out = num_nodes - 1 if output_node is None else output_node
    K = compute_k_blocks(*spec)
    val = mutual_information_from_k(K, output_node=out, input_node=input_node,
                                   jitter=jitter)
    return val / math.log(2.0) if bits else val
