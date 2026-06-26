"""Constraint helpers for movable geometry, plus re-exported power projections.

Power/precoder projections are NOT reimplemented -- they are re-exported from
gaussian-dag. This module adds the geometry-specific constraints a movable-array
study needs: a hard minimum-separation projection and a box (region) projection
for use as ``pga_ascent`` projectors, and a soft differentiable repulsion penalty
to fold into the objective.
"""
from __future__ import annotations

import torch

# Re-export the family's power projections (do not reimplement).
from gaussian_dag import project_frobenius_ball, project_total_power  # noqa: F401


def project_box(center: torch.Tensor, lo, hi) -> torch.Tensor:
    """Clamp an array centre into a box. ``lo``/``hi`` may be scalars or per-axis."""
    c = center.detach().clone()
    lo_t = torch.as_tensor(lo, dtype=c.dtype)
    hi_t = torch.as_tensor(hi, dtype=c.dtype)
    return torch.minimum(torch.maximum(c, lo_t), hi_t)


def project_min_separation(centers, d_min: float, *, iters: int = 50):
    """Push array centres apart until every pair is at least ``d_min`` apart.

    Iterative pairwise correction (Gauss-Seidel style); returns a new list of
    detached centre tensors. Idempotent when the centres are already separated.
    Degenerate coincident pairs are split along the first axis.
    """
    pts = [c.detach().clone() for c in centers]
    n = len(pts)
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[j] - pts[i]
                dist = float(d.norm())
                if dist < d_min - 1e-12:
                    if dist < 1e-9:
                        dirv = torch.zeros_like(pts[i]); dirv[0] = 1.0
                        corr = 0.5 * d_min
                    else:
                        dirv = d / dist
                        corr = 0.5 * (d_min - dist)
                    pts[i] = pts[i] - corr * dirv
                    pts[j] = pts[j] + corr * dirv
                    moved = True
        if not moved:
            break
    return pts


def repulsion_penalty(centers, d_min: float) -> torch.Tensor:
    """Soft, differentiable min-separation penalty (sum of squared violations)."""
    p = torch.zeros((), dtype=centers[0].dtype)
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = ((centers[i] - centers[j]) ** 2).sum()
            p = p + torch.clamp(d_min ** 2 - d2, min=0.0) ** 2
    return p
