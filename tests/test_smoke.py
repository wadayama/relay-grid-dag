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
