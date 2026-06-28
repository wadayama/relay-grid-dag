"""Smoke tests for the precoded engine at realistic scale (health checks).

Distinct from the tie-point tests: these run the engine on full-size / wideband /
edge-case scenes and assert *cheap sane properties* (finite, >= scalar-AF, power
budgets, expected scaling) rather than exact identities. They probe the residual
risks flagged in review: non-convex behavior at full scale, the OFDM path, edge
cases, and a check that the field render uses the supplied precoder.

Run: uv run pytest -q tests/test_smoke_precoded.py
"""
import math

import torch

import relay_grid_dag as rgd
from relay_grid_dag import viz
from relay_grid_dag.precoding import optimize_precoders

P_TX = 100.0
P_RELAY = 100.0


def _relay_power(s, nm, F, W):
    A = s.channel("tx", nm) @ F
    K = A @ A.mH + torch.eye(A.shape[0], dtype=rgd.DTYPE)        # sigma_r = 1
    return float(torch.trace(W @ K @ W.mH).real)


# --- A: full-scale end-to-end (default 8x8, n_relay=4, 25-candidate grid) ----- #
def test_a_fullscale_end_to_end():
    s, names, _ = rgd.build_candidate_scene("near")
    subset = [names[i] for i in (6, 18)]
    scalar_af = rgd.subset_mi(s, subset, P_relay=P_RELAY)
    cap, F, Ws = optimize_precoders(s, "tx", subset, "rx",
                                    P_tx=P_TX, P_relay=P_RELAY, iters=150)
    assert math.isfinite(cap) and cap > scalar_af + 1e-3          # converges, improves
    assert float((F.abs() ** 2).sum()) <= P_TX + 1e-6            # Tx power budget
    for nm, W in zip(subset, Ws):
        assert _relay_power(s, nm, F, W) <= P_RELAY + 1e-4        # relay power budget


# --- B: wideband / per-subcarrier (the OFDM path composes with the engine) ---- #
def test_b_wideband_per_subcarrier():
    s, names, _ = rgd.build_candidate_scene("near")
    subset = [names[i] for i in (6, 18)]
    ks = rgd.subcarrier_wavenumbers(f_c=10.0, S=3, df=0.4)
    for k in ks:
        kf = float(k)
        scalar_af = rgd.subset_mi(s, subset, k_wave=kf, P_relay=P_RELAY)
        cap = optimize_precoders(s, "tx", subset, "rx",
                                 k_wave=kf, P_tx=P_TX, P_relay=P_RELAY, iters=100)[0]
        assert math.isfinite(cap) and cap >= scalar_af - 1e-6


# --- C: edge cases (K=0, single, many, rank-deficient precoder) --------------- #
def test_c_edge_cases():
    s, names, _ = rgd.build_candidate_scene("near")

    c0 = optimize_precoders(s, "tx", [], "rx", iters=100)[0]          # K = 0, direct only
    assert math.isfinite(c0) and c0 > 0

    c1 = optimize_precoders(s, "tx", [names[6]], "rx", iters=100)[0]  # single relay
    assert c1 >= c0 - 1e-6                                            # a relay cannot hurt

    big = [names[i] for i in (1, 6, 11, 16, 21)]                      # many relays
    cbig = optimize_precoders(s, "tx", big, "rx", iters=80)[0]
    assert math.isfinite(cbig) and cbig > 0

    cap_full, _, _ = optimize_precoders(s, "tx", [names[6]], "rx", iters=120)
    cap_d2, Fd2, _ = optimize_precoders(s, "tx", [names[6]], "rx", d=2, iters=120)
    assert Fd2.shape[1] == 2                                          # rank-deficient F
    assert float((Fd2.abs() ** 2).sum()) <= P_TX + 1e-6
    assert math.isfinite(cap_d2) and cap_d2 <= cap_full + 1e-3        # fewer streams <= full


# --- D: the field render uses the supplied precoder ---------------------------- #
def test_d_field_scales_with_precoder():
    """Field intensity is quadratic in F: scaling F by 2 scales intensity by 4.
    A field that ignored F (independent of the precoder) would fail this."""
    s, names, _ = rgd.build_candidate_scene("near", n_tx=4, n_rx=4, n_relay=2)
    n_tx = s.n_ant("tx")
    F0 = math.sqrt(P_TX / n_tx) * torch.eye(n_tx, dtype=rgd.DTYPE)
    grid = s.positions("rx")

    i1 = viz.field_intensity(s, "tx", grid, F=F0, relays=(), relay_mats=[])
    i2 = viz.field_intensity(s, "tx", grid, F=2.0 * F0, relays=(), relay_mats=[])
    assert torch.allclose(i2, 4.0 * i1, rtol=1e-9, atol=1e-9)
    assert torch.isfinite(i1).all() and bool((i1 >= 0).all())

    # activating a relay changes the field (relays are actually rendered)
    nm = names[6]
    n_l = s.n_ant(nm)
    W = math.sqrt(P_RELAY / n_l) * torch.eye(n_l, dtype=rgd.DTYPE)
    i_relay = viz.field_intensity(s, "tx", grid, F=F0, relays=[nm], relay_mats=[W])
    assert not torch.allclose(i_relay, i1)
