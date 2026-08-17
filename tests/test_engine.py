"""Step #2+#3 tie-point tests for the differentiable joint-precoding engine.

These bind the engine (SPEC.md Sec. 4) to ground truths:
  T2 -- AD vs finite differences (validates the whitening + K-recursion chain),
  T4 -- optimized MI >= scalar-AF baseline (empirical; random init),
  T5 -- power budgets hold at the optimizer,
  T6 -- direct-only MI == water-filling on the direct channel,
  T7 -- optimized scalar-AF mode: exact w*I structure, budgets, ordering,
  T8 -- network-total relay power mode: summed budget, per-relay dominance.

Run: uv run pytest -q tests/test_engine.py
"""
import math

import torch

import relay_grid_dag as rgd
from relay_grid_dag.precoding import _merge_mi, optimize_precoders

P_TX = 100.0
P_RELAY = 100.0


def test_t2_ad_vs_finite_difference():
    """Real-parametrized gradients of the merge-channel MI match central finite
    differences --- validates the F-channel, the Cholesky whitening, and the
    de-whitening triangular solve under autograd."""
    s, names, _ = rgd.build_candidate_scene("near", n_tx=2, n_rx=2, n_relay=2)
    subset = [names[6]]
    n_tx, n_l = s.n_ant("tx"), s.n_ant(subset[0])
    torch.manual_seed(0)
    Fr = 1.0 + 0.5 * torch.randn(n_tx, n_tx, dtype=torch.float64)
    Fi = 0.5 * torch.randn(n_tx, n_tx, dtype=torch.float64)
    Wr = 1.0 + 0.3 * torch.randn(n_l, n_l, dtype=torch.float64)
    Wi = 0.3 * torch.randn(n_l, n_l, dtype=torch.float64)

    def mi_of(Fr_, Fi_, Wr_, Wi_):
        F = torch.complex(Fr_, Fi_)
        What = [torch.complex(Wr_, Wi_)]
        return _merge_mi(s, "tx", subset, "rx", F, What,
                         sigma_r=1.0, sigma_d=1.0, k_wave=rgd.K_WAVE)

    Fr_l = Fr.clone().requires_grad_(True)
    Wr_l = Wr.clone().requires_grad_(True)
    mi_of(Fr_l, Fi, Wr_l, Wi).backward()

    eps = 1e-6
    for (i, j) in [(0, 0), (1, 0)]:
        Fp, Fm = Fr.clone(), Fr.clone()
        Fp[i, j] += eps
        Fm[i, j] -= eps
        fd = (float(mi_of(Fp, Fi, Wr, Wi)) - float(mi_of(Fm, Fi, Wr, Wi))) / (2 * eps)
        assert abs(fd - float(Fr_l.grad[i, j])) < 1e-3      # F gradient ~ FD
    for (i, j) in [(0, 0), (1, 1)]:
        Wp, Wm = Wr.clone(), Wr.clone()
        Wp[i, j] += eps
        Wm[i, j] -= eps
        fd = (float(mi_of(Fr, Fi, Wp, Wi)) - float(mi_of(Fr, Fi, Wm, Wi))) / (2 * eps)
        assert abs(fd - float(Wr_l.grad[i, j])) < 1e-3      # whitened-relay grad ~ FD


def test_t4_optimization_improves_on_scalar_af():
    """The optimized MI beats the conventional scalar-AF relay baseline on a
    generic near-field scene (empirical; with random init this holds by a wide
    margin, not by construction)."""
    s, names, _ = rgd.build_candidate_scene("near", n_tx=4, n_rx=4, n_relay=2)
    subset = [names[i] for i in (8, 16)]
    scalar_af = rgd.subset_mi(s, subset, P_relay=P_RELAY)
    opt, F, Ws = optimize_precoders(s, "tx", subset, "rx",
                                    P_tx=P_TX, P_relay=P_RELAY, iters=200)
    assert opt > scalar_af + 1e-3           # precoding beats the scalar-AF baseline


def test_t5_power_budgets_hold():
    """At the optimizer, the Tx and every relay respect their power budgets."""
    s, names, _ = rgd.build_candidate_scene("near", n_tx=4, n_rx=4, n_relay=2)
    subset = [names[i] for i in (8, 16)]
    opt, F, Ws = optimize_precoders(s, "tx", subset, "rx",
                                    P_tx=P_TX, P_relay=P_RELAY, iters=200)
    assert float((F.abs() ** 2).sum()) <= P_TX + 1e-6
    for nm, W in zip(subset, Ws):
        A = s.channel("tx", nm) @ F
        K = A @ A.mH + torch.eye(A.shape[0], dtype=rgd.DTYPE)     # sigma_r = 1
        relay_power = float(torch.trace(W @ K @ W.mH).real)
        assert relay_power <= P_RELAY + 1e-4


def test_multistart_at_least_single():
    """Multi-start (restarts>1) is at least the single (seed-0) start: start 0 is
    shared, and the best over more starts can only improve."""
    s, names, _ = rgd.build_candidate_scene("near", n_tx=4, n_rx=4, n_relay=2)
    subset = [names[i] for i in (8, 16)]
    one = optimize_precoders(s, "tx", subset, "rx", iters=150, restarts=1)[0]
    multi = optimize_precoders(s, "tx", subset, "rx", iters=150, restarts=4)[0]
    assert multi >= one - 1e-6


def test_t6_direct_only_equals_waterfilling():
    """With no relays the engine optimizes the Tx precoder alone; the result equals
    the water-filling capacity of the direct channel and never exceeds it."""
    s, names, _ = rgd.build_candidate_scene("near", n_tx=4, n_rx=4, n_relay=2)
    H_sd = s.channel("tx", "rx", atten=s.direct_atten)
    C = torch.eye(s.n_ant("rx"), dtype=rgd.DTYPE)                 # sigma_d = 1
    wf_cap, _ = rgd.waterfilling_capacity(H_sd, C, P_TX)
    iso_direct = rgd.subset_mi(s, [])                            # isotropic direct MI

    opt = optimize_precoders(s, "tx", [], "rx", P_tx=P_TX, iters=400)[0]
    assert opt <= wf_cap + 1e-6             # cannot beat capacity
    assert opt >= iso_direct - 1e-6         # at least the isotropic input
    assert opt >= wf_cap - 0.05             # reaches the water-filling optimum


def test_t7_optimized_scalar_relay():
    """T7 -- optimized scalar-AF mode (relay_structure="scalar"): the returned
    relay matrices are exact scalar multiples of I, the per-relay power budgets
    hold, and the value sits between the conventional scalar-AF baseline and
    the full-matrix engine."""
    s, names, _ = rgd.build_candidate_scene("near", n_tx=2, n_rx=2, n_relay=2)
    subset = [names[6], names[12]]
    conv = rgd.subset_mi(s, subset)
    v_sc, F, Ws = optimize_precoders(s, "tx", subset, "rx", iters=200,
                                     relay_structure="scalar")
    v_full = optimize_precoders(s, "tx", subset, "rx", iters=200)[0]
    for W in Ws:
        n = W.shape[0]
        off = float((W - W[0, 0] * torch.eye(n, dtype=W.dtype)).abs().max())
        assert off < 1e-12                     # exactly w * I
    for nm, W in zip(subset, Ws):
        A = s.channel("tx", nm) @ F
        K = A @ A.mH + torch.eye(A.shape[0], dtype=A.dtype)
        assert float(torch.trace(W @ K @ W.mH).real) <= P_RELAY * (1 + 1e-9)
    assert v_sc >= conv - 1e-6                 # optimized >= power-normalized
    assert v_full >= v_sc - 1e-6               # matrix >= scalar


def test_t8_total_relay_power():
    """T8 -- network-total relay power mode (power_mode="total"): the summed
    physical relay power respects P_R_tot, and with P_R_tot = P_relay the value
    cannot exceed the per-relay-budget value (the total ball is a subset of the
    per-relay balls)."""
    s, names, _ = rgd.build_candidate_scene("near", n_tx=2, n_rx=2, n_relay=2)
    subset = [names[6], names[12]]
    v_tot, F, Ws = optimize_precoders(s, "tx", subset, "rx", iters=200,
                                      power_mode="total", P_R_tot=P_RELAY)
    v_pr = optimize_precoders(s, "tx", subset, "rx", iters=200)[0]
    tot = 0.0
    for nm, W in zip(subset, Ws):
        A = s.channel("tx", nm) @ F
        K = A @ A.mH + torch.eye(A.shape[0], dtype=A.dtype)
        tot += float(torch.trace(W @ K @ W.mH).real)
    assert tot <= P_RELAY * (1 + 1e-9)
    assert v_tot <= v_pr + 0.1
