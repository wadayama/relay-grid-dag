"""Tie-point test T1: the visualization is consistent with the computation.

The carrier-field render uses the same precoders (F, {W_l}) that produce the
reported MI. T1 binds them: the field intensity summed over the Rx array equals the
model's received signal power tr(G_eff G_eff^H) for the same precoders.

Run: uv run pytest -q tests/test_viz.py
"""
import torch

import relay_grid_dag as rgd
from relay_grid_dag import viz
from relay_grid_dag.precoding import optimize_precoders


def _g_eff(s, subset, F, Ws):
    """Effective stream->output channel at the Rx array for the given precoders."""
    G = s.direct_atten * (s.channel("tx", "rx") @ F)
    for nm, W in zip(subset, Ws):
        G = G + s.channel(nm, "rx") @ (W @ (s.channel("tx", nm) @ F))
    return G


def test_t1_field_at_rx_equals_received_power():
    """Field intensity summed over the Rx elements == tr(G_eff G_eff^H)."""
    s, names, _ = rgd.build_candidate_scene("near", n_tx=3, n_rx=3, n_relay=2)
    subset = [names[8], names[16]]
    _, F, Ws = optimize_precoders(s, "tx", subset, "rx", iters=120)

    rx_pos = s.positions("rx")
    inten = viz.field_intensity(s, "tx", rx_pos, F=F, relays=subset, relay_mats=Ws)

    G = _g_eff(s, subset, F, Ws)
    sig_power = float(torch.trace(G @ G.mH).real)
    assert abs(float(inten.sum()) - sig_power) <= 1e-6 * (1.0 + sig_power)


def test_t1_no_relays_matches_direct_power():
    """With no relays the field at Rx is the direct (blockage-attenuated) power."""
    s, names, _ = rgd.build_candidate_scene("near", n_tx=3, n_rx=3, n_relay=2)
    _, F, Ws = optimize_precoders(s, "tx", [], "rx", iters=120)

    rx_pos = s.positions("rx")
    inten = viz.field_intensity(s, "tx", rx_pos, F=F, relays=(), relay_mats=[])
    G = s.direct_atten * (s.channel("tx", "rx") @ F)
    sig_power = float(torch.trace(G @ G.mH).real)
    assert abs(float(inten.sum()) - sig_power) <= 1e-6 * (1.0 + sig_power)


def test_carrier_field_db_shape_and_peak():
    """The dB map has one value per probe point, peak 0 dB, floored at floor_db."""
    s, names, _ = rgd.build_candidate_scene("near", n_tx=3, n_rx=3, n_relay=2)
    subset = [names[8]]
    _, F, Ws = optimize_precoders(s, "tx", subset, "rx", iters=80)
    # a small probe grid
    import numpy as np
    xs = torch.linspace(0.0, 20.0, 7, dtype=torch.float64)
    ys = torch.linspace(-4.0, 4.0, 5, dtype=torch.float64)
    grid = torch.stack([torch.stack([x, y]) for y in ys for x in xs])
    db = viz.carrier_field(s, "tx", grid, F=F, relays=subset, relay_mats=Ws, floor_db=-40.0)
    assert db.shape == (35,)
    assert abs(db.max()) < 1e-9            # peak normalised to 0 dB
    assert db.min() >= -40.0 - 1e-9        # floored
