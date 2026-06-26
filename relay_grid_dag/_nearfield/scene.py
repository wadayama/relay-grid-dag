"""Named multi-node scene: roles, geometry, and channel extraction.

The analytic counterpart of ``radio-twin``'s ``RadioTwin``. Nodes carry a role
that decides whether they transmit, receive, or both:

    source -> transmitter only
    sink   -> receiver only
    relay  -> co-located transmitter + receiver (active, noisy DAG node)
    ris    -> co-located transmitter + receiver (passive, unit-modulus edge)

The relay-vs-ris *distinction* (active gain matrix vs passive diagonal phase) is
realised downstream in :mod:`nearfield_dag.dag`, exactly as ``radio-twin``
defers it to the objective. ``Scene`` only knows geometry and produces channel
matrices ``H_k`` -- the single boundary at which the gaussian-dag machinery is
fed. Movable nodes keep their centre as an autograd leaf so a mutual-information
objective is differentiable in the array position.
"""
from __future__ import annotations

import torch

from .channel import (
    K_WAVE,
    RDTYPE,
    far_field_channel,
    near_field_channel,
)
from .geometry import movable, ula

_ROLE_TX = {"source": True, "sink": False, "relay": True, "ris": True}
_ROLE_RX = {"source": False, "sink": True, "relay": True, "ris": True}


class Scene:
    """A set of named array nodes over a common near/far-field channel model."""

    def __init__(self, *, model: str = "near", r_ref: float = 10.0,
                 r_min: float = 0.5, spacing: float = 0.5, axis: str = "y",
                 direct_atten: float = 1.0):
        if model not in ("near", "far"):
            raise ValueError(f"model must be 'near' or 'far', got {model!r}")
        self.model = model
        self.r_ref = r_ref
        self.r_min = r_min
        self.spacing = spacing
        self.axis = axis
        self.direct_atten = direct_atten          # default for the direct edge
        self._nodes: dict[str, dict] = {}

    # ------------------------------------------------------------------ build
    def add_node(self, name, center, n, role, *, spacing=None, axis=None,
                 movable_node: bool = False) -> "Scene":
        """Declare a node. ``role`` in {source, sink, relay, ris}.

        If ``movable_node`` is True the centre is stored as an autograd leaf and
        exposed via :attr:`params` for gradient ascent. Returns ``self``.
        """
        if name in self._nodes:
            raise ValueError(f"duplicate node {name!r}")
        if role not in _ROLE_TX:
            raise ValueError(f"unknown role {role!r}")
        c = movable(center) if movable_node else torch.as_tensor(center, dtype=RDTYPE)
        self._nodes[name] = dict(
            center=c, n=int(n), role=role,
            spacing=self.spacing if spacing is None else spacing,
            axis=self.axis if axis is None else axis,
            movable=movable_node,
        )
        return self

    def move(self, name, center) -> "Scene":
        """Reposition a node (re-leafs the centre if the node is movable)."""
        nd = self._nodes[name]
        nd["center"] = movable(center) if nd["movable"] else torch.as_tensor(center, dtype=RDTYPE)
        return self

    # ------------------------------------------------------------- geometry
    def positions(self, name) -> torch.Tensor:
        nd = self._nodes[name]
        return ula(nd["center"], nd["n"], nd["spacing"], nd["axis"])

    def n_ant(self, name) -> int:
        return self._nodes[name]["n"]

    @property
    def params(self) -> list[torch.Tensor]:
        """Movable node centres (autograd leaves), for use as PGA parameters."""
        return [nd["center"] for nd in self._nodes.values() if nd["movable"]]

    # -------------------------------------------------------------- channels
    def channel(self, src, dst, *, atten: float = 1.0,
                k_wave: float = K_WAVE) -> torch.Tensor:
        """Channel ``H`` of shape ``(n_dst, n_src)`` for edge ``src -> dst``.

        Validates that ``src`` can transmit and ``dst`` can receive (role guard),
        then applies the scene's near- or far-field model. ``atten`` scales the
        amplitude (e.g. partial blockage of a direct link). ``k_wave`` is the
        subcarrier wavenumber (default ``2*pi`` = carrier / single-carrier); pass
        ``2*pi*f_i/f_c`` for OFDM subcarrier ``i``. The geometry, and hence the
        distances, are independent of ``k_wave``; only the phase changes.
        """
        for node, table, what in ((src, _ROLE_TX, "transmit"), (dst, _ROLE_RX, "receive")):
            if node not in self._nodes:
                raise ValueError(f"unknown node {node!r}")
            if not table[self._nodes[node]["role"]]:
                raise ValueError(
                    f"node {node!r} (role {self._nodes[node]['role']!r}) cannot {what}")
        tx_pos, rx_pos = self.positions(src), self.positions(dst)
        gen = near_field_channel if self.model == "near" else far_field_channel
        kw = dict(r_ref=self.r_ref, k_wave=k_wave)
        if self.model == "near":
            kw["r_min"] = self.r_min
        return atten * gen(rx_pos, tx_pos, **kw)

    def channels(self, edges) -> dict:
        """Return ``{(src, dst): H}`` for a list of ``(src, dst)`` pairs."""
        return {(s, d): self.channel(s, d) for (s, d) in edges}
