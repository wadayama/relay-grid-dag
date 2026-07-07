"""Regression tests: vendored physics + the scalar-AF baseline (engine-independent).

These pin the physics (near-field channel, MI, gradient) and the conventional
scalar-AF relay baseline (``subset_mi``), which do not depend on the precoder
optimizer. The precoded-MI engine and selection are tested in
``test_engine.py`` / ``test_select_capacity.py`` / ``test_smoke_precoded.py``.

Run: uv run pytest   (or, against an existing env: PYTHONPATH=. pytest)
"""
import itertools

import torch

import relay_grid_dag as rgd


def _greedy_af(s, names, K, P_relay=100.0):
    """Forward greedy on the scalar-AF baseline MI ``subset_mi`` (no optimization)."""
    chosen, remaining = [], set(range(len(names)))
    for _ in range(K):
        best_j, best = None, -1e18
        for j in remaining:
            v = rgd.subset_mi(s, [names[i] for i in chosen + [j]], P_relay=P_relay)
            if v > best:
                best, best_j = v, j
        chosen.append(best_j); remaining.discard(best_j)
    return sorted(chosen)


def _oracle_af(s, names, K, P_relay=100.0):
    """Exhaustive best K-subset on the scalar-AF baseline MI -> (indices, mi)."""
    best_set, best = None, -1e18
    for c in itertools.combinations(range(len(names)), K):
        v = rgd.subset_mi(s, [names[i] for i in c], P_relay=P_relay)
        if v > best:
            best, best_set = v, c
    return sorted(best_set), best


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
    """Scalar-AF baseline: MI-aware greedy beats received-power, is near the
    exhaustive oracle, and K=0 reduces to the direct link. (Precoded-MI
    selection is tested in test_select_capacity.)"""
    s, names, coords = rgd.build_candidate_scene("near")
    direct = rgd.direct_only_mi(s)
    assert rgd.subset_mi(s, []) == direct                      # K=0 == direct

    K = 3
    greedy = rgd.subset_mi(s, [names[i] for i in _greedy_af(s, names, K)])
    recv = rgd.subset_mi(s, [names[i] for i in rgd.select_received_power(s, names, K)])
    assert greedy > recv + 0.2                                 # smart beats naive
    assert greedy > direct                                     # relays help

    o_idx, o_mi = _oracle_af(s, names, 2)
    g2 = rgd.subset_mi(s, [names[i] for i in _greedy_af(s, names, 2)])
    assert o_mi + 1e-6 >= g2 >= 0.95 * o_mi                    # greedy near-oracle


def test_near_field_higher_rank_than_far():
    """Near-field single-relay MI exceeds the far-field (rank-deficient) link."""
    sn, names, _ = rgd.build_candidate_scene("near")
    sf, namesf, _ = rgd.build_candidate_scene("far")
    best_near = max(rgd.subset_mi(sn, [nm]) for nm in names)
    best_far = max(rgd.subset_mi(sf, [nm]) for nm in namesf)
    assert best_near > best_far


def test_t3_far_field_rank_one_near_field_high_rank():
    """T3 (SPEC.md §8): the far-field direct channel collapses to rank one, while
    the near-field channel is genuinely high-rank at the same geometry."""
    sn, _, _ = rgd.build_candidate_scene("near")
    sf, _, _ = rgd.build_candidate_scene("far")
    H_near = sn.channel("tx", "rx", atten=sn.direct_atten)
    H_far = sf.channel("tx", "rx", atten=sf.direct_atten)
    s_far = torch.linalg.svdvals(H_far).double()
    s_near = torch.linalg.svdvals(H_near).double()
    # far field: a single dominant singular value (numerically rank one)
    assert float(s_far[1] / s_far[0]) < 1e-6
    assert rgd.viz.effrank(H_far) < 1.01
    # near field at the same 20-lambda link: effective rank clearly above one
    assert rgd.viz.effrank(H_near) > 1.2
    assert float(s_near[1] / s_near[0]) > 1e-3


def test_continuous_to_discrete_recovers_selection():
    """continuous position PGA -> grid rounding lands at a strong on-grid set
    (scored by the scalar-AF baseline MI)."""
    s, names, coords = rgd.build_candidate_scene("near")
    final, _ = rgd.continuous_relays(2, [6.0, -4.0], [14.0, 4.0], starts=3)
    ridx = rgd.round_to_grid(final, coords)
    r_mi = rgd.subset_mi(s, [names[i] for i in ridx])
    _, o_mi = _oracle_af(s, names, 2)
    assert r_mi >= 0.9 * o_mi


def test_wideband_per_subcarrier():
    """OFDM path: a shared set has a finite, band-varying per-subcarrier rate."""
    s, names, _ = rgd.build_candidate_scene("near")
    S = [names[i] for i in rgd.select_received_power(s, names, 3)]
    k_waves = rgd.subcarrier_wavenumbers(f_c=10.0, S=8, df=0.4)
    rates = [rgd.subset_mi(s, S, k_wave=k) for k in k_waves]
    assert all(r > 0 for r in rates) and len(rates) == 8
    assert max(rates) > min(rates)                       # beam squint varies the rate


def test_naive_baselines_and_k0():
    """Distance baseline is well-formed; K=0 reduces every selector to the direct link."""
    s, names, coords = rgd.build_candidate_scene("near")
    di = rgd.select_distance(coords, 3)
    assert len(di) == 3 and len(set(di)) == 3 and all(0 <= i < len(names) for i in di)
    assert rgd.select_greedy(s, names, 0) == []               # K=0 -> empty set
    assert rgd.select_received_power(s, names, 0) == []
    assert rgd.select_distance(coords, 0) == []
    assert rgd.subset_mi(s, []) == rgd.direct_only_mi(s)


def test_p_relay_plumbed_through_baseline():
    """The P_relay knob is honored end to end (subset_mi and the scalar-AF oracle)."""
    s, names, _ = rgd.build_candidate_scene("near")
    sub = [names[6], names[17]]
    assert rgd.subset_mi(s, sub, P_relay=1000.0) > rgd.subset_mi(s, sub, P_relay=10.0)
    _, mi_lo = _oracle_af(s, names, 2, P_relay=10.0)
    _, mi_hi = _oracle_af(s, names, 2, P_relay=1000.0)
    assert mi_hi > mi_lo


def test_input_validation():
    """Selectors and the baseline reject nonsensical inputs with clear errors,
    instead of returning wrong results or failing deep in the K-recursion."""
    import pytest

    s, names, coords = rgd.build_candidate_scene("near", coords=rgd.grid_coords(2, 1))
    assert len(names) == 2
    # K > L: greedy and exhaustive both raise (not NoneType crashes)
    with pytest.raises(ValueError):
        rgd.select_greedy(s, names, 3, iters=5)
    with pytest.raises(ValueError):
        rgd.select_exhaustive(s, names, 3, iters=5)
    # duplicate relays in a subset are rejected (silent double-counting otherwise)
    with pytest.raises(ValueError):
        rgd.subset_mi(s, [names[0], names[0]])
    with pytest.raises(ValueError):
        rgd.optimize_precoders(s, "tx", [names[0], names[0]], "rx", iters=5)
    # non-positive power budgets are rejected
    with pytest.raises(ValueError):
        rgd.optimize_precoders(s, "tx", [names[0]], "rx", P_relay=-1.0, iters=5)


def test_multipair_received_power_k0_and_kgtl():
    """The multi-pair baseline guards K=0 (not the whole grid) and K>L raises."""
    import pytest

    s, names, _ = rgd.build_pair_scene(coords=rgd.grid_coords(2, 1))
    assert rgd.received_power_pairs(s, 2, names, 0) == []       # K=0 -> empty, not all
    with pytest.raises(ValueError):
        rgd.select_greedy_sumrate(s, 2, names, 5)
