"""Field / power / rank diagnostics (array producers; plotting lives in examples).

These return plain numpy/tensor quantities so the library carries no plotting
dependency; the example scripts (which require the ``examples`` extra) do the
matplotlib rendering. The three quantities expose *why* mutual information moves
as geometry changes: the carrier field (where energy goes in space, computed from
the supplied Tx/relay precoders -- the same ones that produce the MI), the
received power, and the effective rank of the effective channel.
"""
from __future__ import annotations

import numpy as np
import torch

from .channel import DTYPE, far_field_channel, near_field_channel


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


def field_intensity(scene, src, grid, *, F, relays=(), relay_mats=None) -> torch.Tensor:
    """Expected signal intensity ``||g(p)||^2`` at each probe point ``p`` for the
    given precoders, with the source streams ``X ~ CN(0, I)``:

        g(p) = atten * h(p, src) F  +  sum_l h(p, relay_l) W_l H_sr,l F,

    where ``atten = scene.direct_atten`` is the same blockage factor the MI puts on
    the direct Tx->Rx link, and ``W_l = relay_mats[l]`` are the physical relay
    matrices. The field is built from the supplied precoders ``(F, {W_l})`` and the
    scene channels --- the same quantities that produce the mutual information ---
    so the field and the MI describe one transmit configuration (SPEC.md Sec. 6).
    Evaluated at the Rx array, ``sum_pixels`` of this equals ``tr(G_eff G_eff^H)``
    (the model received signal power); the tie-point test T1 checks this. The
    probe channel follows ``scene.model`` (near or far), the same physics as the
    MI. Returns a linear-power tensor of shape ``(n_pix,)``.
    """
    def chan(rx_pos, tx_pos):
        if scene.model == "near":
            return near_field_channel(rx_pos, tx_pos, r_ref=scene.r_ref,
                                      r_min=scene.r_min)
        return far_field_channel(rx_pos, tx_pos, r_ref=scene.r_ref)

    with torch.no_grad():
        tx = scene.positions(src)
        G = scene.direct_atten * (chan(grid, tx) @ F)
        for i, relay in enumerate(relays):
            rly = scene.positions(relay)
            G = G + chan(grid, rly) @ (relay_mats[i].to(DTYPE) @ (chan(rly, tx) @ F))
        return (G.abs() ** 2).sum(dim=1)


def carrier_field(scene, src, grid, *, F, relays=(), relay_mats=None,
                  floor_db=-40.0) -> np.ndarray:
    """Carrier signal field in dB at the points ``grid`` (shape ``(n_pix, D)``),
    computed from the supplied precoders ``(F, {W_l})`` via :func:`field_intensity`
    (the same precoders that produce the reported MI; SPEC.md Sec. 6), normalised to
    its peak (0 dB) and floored at ``floor_db``. The field focuses energy toward the
    destination to the extent that the supplied ``(F, {W_l})`` do."""
    inten = field_intensity(scene, src, grid, F=F, relays=relays,
                            relay_mats=relay_mats).numpy()
    db = 10.0 * np.log10(inten / inten.max() + 1e-12)
    return np.clip(db, floor_db, 0.0)
