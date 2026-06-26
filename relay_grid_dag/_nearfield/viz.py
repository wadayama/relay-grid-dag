"""Field / power / rank diagnostics (array producers; plotting lives in examples).

These return plain numpy/tensor quantities so the library carries no plotting
dependency; the example scripts (which require the ``examples`` extra) do the
matplotlib rendering. The three quantities expose *why* mutual information moves
as geometry changes: the carrier field (where energy goes in space), the received
power, and the effective rank of the effective channel.
"""
from __future__ import annotations

import numpy as np
import torch

from .channel import DTYPE, near_field_channel


def rxpow(H: torch.Tensor, P: float) -> float:
    """Isotropic received signal power ``tr(H Sigma_X H^H) = (P/n_tx)||H||_F^2``."""
    n_tx = H.shape[1]
    return float((P / n_tx) * H.abs().pow(2).sum())


def effrank(H: torch.Tensor) -> float:
    """Effective rank of ``H``: participation ratio of its squared singular values.

    ``(sum s_i^2)^2 / sum s_i^4`` -- 1 for a rank-one (single-mode) channel, up to
    ``rank(H)`` when the modes are balanced. A scalar measure of spatial DoF.
    """
    e = torch.linalg.svdvals(H).double() ** 2
    return float(e.sum() ** 2 / e.pow(2).sum())


def carrier_field(scene, src, dst, grid, *, relays=(), P_tx=100.0,
                  P_relay=100.0, sigma_r=1.0, floor_db=-40.0) -> np.ndarray:
    """Total carrier field |E| in dB at the points ``grid`` (shape ``(n_pix, D)``).

    The source array is conjugate-beamformed at the ``dst`` centroid (matched to
    the direct LoS channel, power ``P_tx``); each relay re-radiates the carrier it
    receives, amplified by a scalar AF gain normalised to ``P_relay``:

        E(p) = h(p, src) w  +  sum_i h(p, relay_i) (g_i H_sr_i w).

    Returned in dB normalised to the peak (0 dB), floored at ``floor_db``. Uses
    the scene's ``r_ref``/``r_min``. (Near-field model.)
    """
    rr, rm = scene.r_ref, scene.r_min
    with torch.no_grad():
        tx = scene.positions(src)
        dst_c = scene.positions(dst).mean(dim=0, keepdim=True)
        w = torch.conj(near_field_channel(dst_c, tx, r_ref=rr, r_min=rm)[0])
        w = w * torch.sqrt(torch.tensor(P_tx, dtype=torch.float64)
                           / (w.abs() ** 2).sum()).to(DTYPE)
        E = near_field_channel(grid, tx, r_ref=rr, r_min=rm) @ w
        for relay in relays:
            rly = scene.positions(relay)
            recv = near_field_channel(rly, tx, r_ref=rr, r_min=rm) @ w
            g = torch.sqrt(P_relay / (recv.abs() ** 2).sum().clamp(min=1e-12))
            E = E + near_field_channel(grid, rly, r_ref=rr, r_min=rm) @ (g * recv)
        A = E.abs().numpy()
        db = 20.0 * np.log10(A / A.max() + 1e-12)
        return np.clip(db, floor_db, 0.0)
