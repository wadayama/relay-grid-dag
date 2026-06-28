"""Step #4 (T7) + #5 tests: f(S) monotonicity and selection on the MI oracle.

  T7 -- f(S u {l}) >= f(S): adding a relay never lowers the optimized MI
        (free disposal, SPEC.md Prop. 2). Shown by warm-starting the larger set at
        the smaller set's optimum with the new relay off (W = 0).
  #5 -- greedy on the MI oracle is near the exhaustive oracle and is at
        least as good as the received-power baseline (scored under MI).

Run: uv run pytest -q tests/test_select_capacity.py
"""
import torch

import relay_grid_dag as rgd
from relay_grid_dag.precoding import optimize_precoders

P_TX = 100.0
P_RELAY = 100.0


def _small_scene():
    coords = rgd.grid_coords(nx=3, ny=2)          # 6 candidates
    return rgd.build_candidate_scene("near", coords=coords, n_tx=3, n_rx=3, n_relay=2)


def test_t7_capacity_monotone_free_disposal():
    """f(S u {l}) >= f(S): warm-start the larger set from S's optimum with the new
    relay off, so it starts at f(S) and can only improve."""
    s, names, _ = _small_scene()
    S = [names[1], names[4]]
    cap_S, F_S, W_S = optimize_precoders(s, "tx", S, "rx",
                                         P_tx=P_TX, P_relay=P_RELAY, iters=150)

    new = names[2]
    n_c = s.n_ant(new)
    W_init = list(W_S) + [torch.zeros(n_c, n_c, dtype=rgd.DTYPE)]   # new relay off
    cap_Sc = optimize_precoders(s, "tx", S + [new], "rx",
                                P_tx=P_TX, P_relay=P_RELAY,
                                F_init=F_S, W_init=W_init, iters=150)[0]
    assert cap_Sc >= cap_S - 1e-6


def test_greedy_capacity_near_oracle_and_beats_baseline():
    """Greedy on the optimized MI is near the exhaustive oracle and at least
    matches the received-power baseline scored under the same MI oracle."""
    s, names, _ = _small_scene()
    K = 2
    kw = dict(P_tx=P_TX, P_relay=P_RELAY, iters=120)

    g_idx = rgd.select_greedy(s, names, K, **kw)
    o_idx, o_cap = rgd.select_exhaustive(s, names, K, **kw)
    g_cap = optimize_precoders(s, "tx", [names[i] for i in g_idx], "rx", **kw)[0]
    assert o_cap + 1e-6 >= g_cap >= 0.95 * o_cap          # greedy near-oracle

    rp_idx = rgd.select_received_power(s, names, K)
    rp_cap = optimize_precoders(s, "tx", [names[i] for i in rp_idx], "rx", **kw)[0]
    assert g_cap >= rp_cap - 1e-6                          # >= received-power baseline


def test_warm_start_matches_cold_start():
    """Greedy with warm start reaches essentially the same MI as cold start
    (warm start is a speedup, not a different answer) on a small scene."""
    s, names, _ = _small_scene()
    kw = dict(P_tx=P_TX, P_relay=P_RELAY, iters=150)
    warm = rgd.select_greedy(s, names, 2, warm_start=True, **kw)
    cold = rgd.select_greedy(s, names, 2, warm_start=False, **kw)
    cap_warm = optimize_precoders(s, "tx", [names[i] for i in warm], "rx", **kw)[0]
    cap_cold = optimize_precoders(s, "tx", [names[i] for i in cold], "rx", **kw)[0]
    assert abs(cap_warm - cap_cold) < 0.05
