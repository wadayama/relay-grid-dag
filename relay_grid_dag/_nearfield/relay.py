"""Amplify-and-forward relay helpers: gain normalisation, channel collapse,
and a closed-form water-filling reference.

These wrap the spherical-wave channels into the quantities a relay study needs:
a power-normalised scalar AF gain, the effective Tx-antenna -> Rx channel of a
fixed-gain diamond (direct + relayed merge), and the exact precoder optimum
(water-filling) used as a correctness oracle. All MI evaluation/optimization is
delegated to gaussian-dag elsewhere; this module only produces channel-level
quantities.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .channel import DTYPE, K_WAVE


def af_gain(H_sr: torch.Tensor, input_cov: torch.Tensor,
            sigma_r: float, P_relay: float) -> torch.Tensor:
    """Scalar AF gain ``g`` normalising relay output power to ``P_relay``.

    With received relay covariance ``K_11 = H_sr Sigma_X H_sr^H + sigma_r^2 I``,
    returns ``g = sqrt(P_relay / tr(K_11))`` as a (differentiable) tensor. A
    frozen ``g`` makes the diamond linear in the precoder, so water-filling is the
    exact precoder optimum.
    """
    n_r = H_sr.shape[0]
    Sig_r = (sigma_r ** 2) * torch.eye(n_r, dtype=DTYPE)
    K_11 = H_sr @ input_cov @ H_sr.mH + Sig_r
    return torch.sqrt(P_relay / torch.trace(K_11).real)


def physical_channel(scene, src, relay, dst, g, *, sigma_r: float = 1.0,
                     sigma_d: float = 1.0, k_wave: float = K_WAVE
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse a fixed-gain diamond to ``Y = G_phys X + R``, ``R ~ CN(0, C_eff)``.

    ``G_phys = H_sd + H_rd (g I) H_sr`` (direct + relayed), and
    ``C_eff = g^2 H_rd Sig_r H_rd^H + Sig_d``. Both are differentiable in the
    node positions. The direct link uses ``scene.direct_atten``. ``k_wave``
    selects the subcarrier (default carrier / single-carrier).
    """
    H_sd = scene.channel(src, dst, atten=scene.direct_atten, k_wave=k_wave)
    H_sr = scene.channel(src, relay, k_wave=k_wave)
    H_rd = scene.channel(relay, dst, k_wave=k_wave)
    n_r, n_d = scene.n_ant(relay), scene.n_ant(dst)
    Sig_r = (sigma_r ** 2) * torch.eye(n_r, dtype=DTYPE)
    Sig_d = (sigma_d ** 2) * torch.eye(n_d, dtype=DTYPE)
    G_phys = H_sd + g * (H_rd @ H_sr)
    C_eff = (g ** 2) * (H_rd @ Sig_r @ H_rd.mH) + Sig_d
    return G_phys, C_eff


def waterfilling_capacity(G_phys: torch.Tensor, C_eff: torch.Tensor,
                          P: float) -> tuple[float, np.ndarray]:
    """Closed-form precoder optimum for ``Y = G X + N``, ``N ~ CN(0, C)``,
    ``tr(Cov(X)) <= P``.

    Whitens by ``C = L L^H`` and water-fills over the singular values of
    ``L^{-1} G``. Returns ``(capacity_bits, power_allocation)``. No autograd
    (reference oracle).
    """
    with torch.no_grad():
        L = torch.linalg.cholesky(C_eff)
        Hw = torch.linalg.solve(L, G_phys)              # whitened channel
        gain = torch.linalg.svdvals(Hw).double() ** 2   # eigenvalues of Hw^H Hw
        inv = 1.0 / gain.clamp(min=1e-15)
        lo, hi = inv.min().item(), inv.max().item() + P
        for _ in range(200):                            # bisection on water level
            mu = 0.5 * (lo + hi)
            tot = torch.clamp(mu - inv, min=0.0).sum().item()
            lo, hi = (mu, hi) if tot < P else (lo, mu)
        p = torch.clamp(mu - inv, min=0.0)
        cap = torch.log2(1.0 + gain * p).sum().item()
        return cap, p.numpy()
