"""Array geometry primitives: element positions and movable (autograd) centres.

A node's geometry is a small array (a ULA in v0.1) described by its centre, the
number of elements, the spacing, and the axis along which the elements lie. The
centre may be a plain coordinate or a torch leaf tensor; when it is a leaf, the
element positions stay in the autograd graph, so a downstream mutual-information
objective is differentiable in the array centre -- this is how movable antennas /
movable relays are optimized.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .channel import RDTYPE


def ula(center, n: int, spacing: float = 0.5, axis: str = "y") -> torch.Tensor:
    """Uniform linear array element positions, shape ``(n, D)``.

    ``center`` may be a sequence ``(x, y[, z])`` or a torch tensor; if it is a
    leaf with ``requires_grad=True`` the returned positions stay in the autograd
    graph so the whole array can be moved differentiably. ``axis`` selects the
    line direction ("x" or "y"); ``D`` is taken from ``center`` (2 by default).
    """
    if not torch.is_tensor(center):
        center = torch.tensor(center, dtype=RDTYPE)
    D = center.shape[-1]
    offs = (torch.arange(n, dtype=RDTYPE) - (n - 1) / 2.0) * spacing
    delta = torch.zeros(n, D, dtype=RDTYPE)
    col = {"x": 0, "y": 1, "z": 2}[axis]
    delta[:, col] = offs
    return center.reshape(1, D) + delta


def movable(center, requires_grad: bool = True) -> torch.Tensor:
    """Return a real leaf tensor for an array centre, ready for gradient ascent.

    Convenience wrapper so callers can write ``movable([8.0, 0.0])`` and pass the
    result both as an array centre and as a PGA parameter.
    """
    t = torch.as_tensor(center, dtype=RDTYPE).clone()
    return t.requires_grad_(requires_grad)


@dataclass
class ArrayGeometry:
    """Declarative ULA geometry container (mirrors the family's geometry pattern).

    Holds the centre (possibly a leaf), element count, spacing and axis, and
    materialises element positions via :meth:`positions`. This is the seam where
    a UPA / 3-D array will be added later without changing call sites.
    """

    center: torch.Tensor
    n: int
    spacing: float = 0.5
    axis: str = "y"

    def positions(self) -> torch.Tensor:
        return ula(self.center, self.n, self.spacing, self.axis)
