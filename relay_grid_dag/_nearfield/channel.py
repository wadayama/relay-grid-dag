"""Analytic near-field (spherical-wave) and far-field (planar-wave) LoS channels.

Geometry is in units of the wavelength (lambda = 1, k = 2*pi). Positions are
``(..., D)`` real tensors; ``D = 2`` is shipped and tested, but every routine is
written over the last axis so ``D = 3`` drops in unchanged.

The near-field LoS channel between element positions ``q_n`` (tx) and ``p_m`` (rx)
follows the spherical-wave model of the project notes,

    [H]_{m,n} = (r_ref / r_{m,n}) * exp(-j k r_{m,n}),   r_{m,n} = ||p_m - q_n||,

with the per-pair distance driving both the amplitude taper and the phase. The
far-field channel is the planar-wave limit (one range to the array centroid plus
a linear phase across the aperture) and is provided as the baseline for
near-field-vs-far-field comparisons.

These functions are the only place the electromagnetic model lives; everything
downstream (DAG assembly, mutual information) treats the result as a plain
complex matrix. They are fully differentiable in the element positions via
PyTorch complex autograd.
"""
from __future__ import annotations

import math

import torch

DTYPE = torch.complex128
RDTYPE = torch.float64
K_WAVE = 2.0 * math.pi          # wavenumber for lambda = 1


def distances(rx_pos: torch.Tensor, tx_pos: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances, shape ``(n_rx, n_tx)``.

    ``rx_pos`` is ``(n_rx, D)`` and ``tx_pos`` is ``(n_tx, D)``; the norm is taken
    over the last (coordinate) axis, so any ``D`` works. Differentiable in both
    position tensors.
    """
    diff = rx_pos[:, None, :] - tx_pos[None, :, :]      # (n_rx, n_tx, D)
    return torch.linalg.vector_norm(diff, dim=-1)       # (n_rx, n_tx)


def near_field_channel(
    rx_pos: torch.Tensor,
    tx_pos: torch.Tensor,
    r_ref: float = 10.0,
    r_min: float = 0.5,
    k_wave: float = K_WAVE,
) -> torch.Tensor:
    """Spherical-wave LoS channel ``H`` of shape ``(n_rx, n_tx)``.

    The amplitude is normalised as ``r_ref / r`` so a pair at distance ``r_ref``
    has unit gain (this only sets the overall SNR scale and leaves the near-field
    phase geometry untouched). The distance is floored at ``r_min``: the
    spherical-wave model is invalid below ~lambda/2 and the floor keeps ``1/r``
    finite if two elements coincide. ``k_wave`` is the subcarrier wavenumber
    (``2*pi`` at the carrier, ``2*pi*f_i/f_c`` on OFDM subcarrier ``i``); only the
    phase depends on it, the distances and amplitude taper do not. Differentiable
    in the positions.
    """
    r = distances(rx_pos, tx_pos).clamp(min=r_min)
    phase = torch.exp(torch.complex(torch.zeros_like(r), -k_wave * r))
    return r_ref * phase / r.to(DTYPE)


def far_field_channel(
    rx_pos: torch.Tensor,
    tx_pos: torch.Tensor,
    r_ref: float = 10.0,
    k_wave: float = K_WAVE,
) -> torch.Tensor:
    """Planar-wave (far-field) LoS channel ``H`` of shape ``(n_rx, n_tx)``.

    The baseline counterpart to :func:`near_field_channel`: the amplitude is the
    single centroid-to-centroid distance (``r_ref / r_c``, constant across the
    aperture) and the phase keeps only the first-order (linear) term of the
    Taylor expansion about the array centroids,

        r_{m,n} ~= r_c + u^T (p_m - q_m_c) - u^T (q_n - q_n_c),

    with ``u`` the unit vector between centroids. This is the standard plane-wave
    model: amplitude becomes range-independent across the array and only the
    linear phase (angle) survives, so a single-path far-field LoS link is
    rank-one. Use it to contrast with the high-rank near-field channel.
    """
    rx_c = rx_pos.mean(dim=0)
    tx_c = tx_pos.mean(dim=0)
    sep = rx_c - tx_c
    r_c = torch.linalg.vector_norm(sep)
    u = sep / r_c.clamp(min=1e-12)                      # propagation direction
    # First-order phase: r_c + u . (p_m - rx_c) - u . (q_n - tx_c).
    rx_proj = (rx_pos - rx_c) @ u                       # (n_rx,)
    tx_proj = (tx_pos - tx_c) @ u                       # (n_tx,)
    r_lin = r_c + rx_proj[:, None] - tx_proj[None, :]   # (n_rx, n_tx)
    phase = torch.exp(torch.complex(torch.zeros_like(r_lin), -k_wave * r_lin))
    return (r_ref / r_c.to(DTYPE)) * phase
