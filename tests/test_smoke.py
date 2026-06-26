"""Regression tests for relay-grid-dag: vendored physics + selection layer.

Run: uv run pytest   (or, against an existing env: PYTHONPATH=. pytest)
"""
import numpy as np
import torch

import relay_grid_dag as rgd


def test_physics_diamond_mi_and_gradient():
    """Vendored near-field engine: diamond MI is finite/positive and gives a
    native position gradient (the gaussian-dag boundary works)."""
    s = rgd.Scene(model="near", direct_atten=0.3)
    s.add_node("tx", [0.0, 0.0], 8, "source")
    s.add_node("rx", [20.0, 0.0], 8, "sink")
    s.add_node("relay", [8.0, 0.0], 4, "relay", movable_node=True)
    I = rgd.mi(rgd.diamond(s, "tx", "relay", "rx"))
    assert torch.isfinite(I) and I.item() > 0
    I.backward()
    g = s.params[0].grad
    assert g is not None and torch.isfinite(g).all()


def test_selection_value():
    """MI-aware selection beats the naive received-power baseline; greedy is
    near the exhaustive oracle; K=0 reduces to the direct link."""
    s, names, coords = rgd.build_candidate_scene("near")
    direct = rgd.direct_only_mi(s)
    assert rgd.subset_mi(s, []) == direct                      # K=0 == direct

    K = 3
    greedy = rgd.subset_mi(s, [names[i] for i in rgd.select_greedy_mi(s, names, K)])
    recv = rgd.subset_mi(s, [names[i] for i in rgd.select_received_power(s, names, K)])
    assert greedy > recv + 0.2                                 # smart beats naive
    assert greedy > direct                                     # relays help

    o_idx, o_mi = rgd.select_exhaustive(s, names, 2)
    g2 = rgd.subset_mi(s, [names[i] for i in rgd.select_greedy_mi(s, names, 2)])
    assert o_mi + 1e-6 >= g2 >= 0.95 * o_mi                    # greedy near-oracle


def test_near_field_higher_rank_than_far():
    """Near-field single-relay MI exceeds the far-field (rank-deficient) link."""
    sn, names, _ = rgd.build_candidate_scene("near")
    sf, namesf, _ = rgd.build_candidate_scene("far")
    best_near = max(rgd.subset_mi(sn, [nm]) for nm in names)
    best_far = max(rgd.subset_mi(sf, [nm]) for nm in namesf)
    assert best_near > best_far


def test_continuous_to_discrete_recovers_selection():
    """continuous PGA -> grid rounding -> 1-swap lands at a strong on-grid set."""
    s, names, coords = rgd.build_candidate_scene("near")
    final, _ = rgd.continuous_relays(2, [6.0, -4.0], [14.0, 4.0], starts=3)
    ridx = rgd.round_to_grid(final, coords)
    sidx, r_swap = rgd.swap_search(s, names, ridx, 2)
    _, o_mi = rgd.select_exhaustive(s, names, 2)
    assert r_swap >= 0.97 * o_mi


def test_wideband_per_subcarrier():
    """OFDM path: a shared set has a finite, band-varying per-subcarrier rate."""
    s, names, _ = rgd.build_candidate_scene("near")
    S = [names[i] for i in rgd.select_greedy_mi(s, names, 3)]
    k_waves = rgd.subcarrier_wavenumbers(f_c=10.0, S=8, df=0.4)
    rates = [rgd.subset_mi(s, S, k_wave=k) for k in k_waves]
    assert all(r > 0 for r in rates) and len(rates) == 8
    assert max(rates) > min(rates)                       # beam squint varies the rate


def test_naive_baselines_and_k0():
    """Distance baseline is well-formed; K=0 reduces every selector to the direct link."""
    s, names, coords = rgd.build_candidate_scene("near")
    di = rgd.select_distance(coords, 3)
    assert len(di) == 3 and len(set(di)) == 3 and all(0 <= i < len(names) for i in di)
    # K=0: every selector returns the empty set, whose MI is the direct link.
    assert rgd.select_greedy_mi(s, names, 0) == []
    assert rgd.select_received_power(s, names, 0) == []
    assert rgd.select_distance(coords, 0) == []
    assert rgd.subset_mi(s, []) == rgd.direct_only_mi(s)


def test_p_relay_plumbed_through_selectors():
    """The P_relay knob is honored consistently (subset_mi, greedy, exhaustive, swap)."""
    s, names, _ = rgd.build_candidate_scene("near")
    sub = [names[6], names[17]]
    lo = rgd.subset_mi(s, sub, P_relay=10.0)
    hi = rgd.subset_mi(s, sub, P_relay=1000.0)
    assert hi > lo                                       # more relay power -> more MI
    # exhaustive must use the passed P_relay, not the default, end to end.
    _, mi_lo = rgd.select_exhaustive(s, names, 2, P_relay=10.0)
    _, mi_hi = rgd.select_exhaustive(s, names, 2, P_relay=1000.0)
    assert mi_hi > mi_lo
