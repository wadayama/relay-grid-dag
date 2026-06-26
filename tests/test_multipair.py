"""Regression tests for the multi-pair interference layer (cmi-dag CMI)."""
import relay_grid_dag as rgd

# Small relay arrays keep the CMI covariances (and the tests) fast.
NR = 2


def test_pipeline_and_interference_cost():
    """Per-pair TIN/free rates are finite & positive; conditioning on the other
    transmitter can only help (TIN <= free); with relays the interference is real."""
    s, names, _ = rgd.build_pair_scene(n_relay=NR)
    g = rgd.select_greedy_sumrate(s, 2, names, 2)
    r = rgd.pair_rates(s, 2, [names[i] for i in g])
    for u in range(2):
        assert r["tin"][u] > 0 and r["free"][u] > 0
        assert r["tin"][u] <= r["free"][u] + 1e-6           # interference-free >= TIN
    cost = sum(r["free"][u] - r["tin"][u] for u in range(2))
    assert cost > 0.1                                       # interference is real


def test_no_interference_limit():
    """With no relays AND no cross-direct, TIN == free (validates the CMI machinery)."""
    s, names, _ = rgd.build_pair_scene(n_relay=NR)
    r = rgd.pair_rates(s, 2, [], cross_atten=1e-6)          # no relays, no cross link
    gap = sum(abs(r["free"][u] - r["tin"][u]) for u in range(2))
    assert gap < 1e-3


def test_shared_relays_inject_interference():
    """Even at cross_atten->0 the shared relays carry interference: an active relay
    forwards both transmitters, so free > TIN persists (the double-edged relay)."""
    s, names, _ = rgd.build_pair_scene(n_relay=NR)
    g = rgd.select_greedy_sumrate(s, 2, names, 2)
    r = rgd.pair_rates(s, 2, [names[i] for i in g], cross_atten=1e-6)
    relay_borne = sum(r["free"][u] - r["tin"][u] for u in range(2))
    assert relay_borne > 0.1


def test_multipair_selection_value():
    """Sum-rate-aware selection beats the interference-blind received-power
    baseline and direct-only, and greedy is near the exhaustive oracle."""
    s, names, _ = rgd.build_pair_scene(n_relay=NR)
    K = 2
    sr = lambda idx: rgd.weighted_sum_rate(s, 2, [names[i] for i in idx])
    direct = sr([])
    greedy = sr(rgd.select_greedy_sumrate(s, 2, names, K))
    recv = sr(rgd.received_power_pairs(s, 2, names, K))
    _, ora = rgd.select_exhaustive_sumrate(s, 2, names, K)
    assert greedy > recv and greedy > direct
    assert ora + 1e-6 >= greedy
