"""Smoke tests for the multi-pair precoded engine (multipair_precoding.py):
optimize {F_u},{W_l} for the 2-pair TIN objective via the cmi-dag conditional MI.

Checks: the optimizer runs and beats the scalar-AF multipair baseline, power
budgets hold at the optimizer, min-rate works, and multi-start >= single start.

Run: uv run pytest -q tests/test_multipair_precoding.py
"""
import math

import torch

import relay_grid_dag.multipair as mp
from relay_grid_dag.multipair_precoding import optimize_multipair, _relay_chols_mp


def _scene(active_idx=(5, 10)):
    scene, names, _ = mp.build_pair_scene()
    return scene, [names[i] for i in active_idx]


def test_optimize_multipair_beats_scalar_af():
    """Optimized sum-rate (matrix relays + Tx precoders) exceeds the scalar-AF
    multipair baseline; per-pair rates are finite and positive."""
    scene, active = _scene()
    base = sum(mp.pair_rates(scene, 2, active)["tin"])             # scalar-AF baseline
    sr, F, W, tin = optimize_multipair(scene, 2, active, objective="sumrate",
                                       iters=150, restarts=2)
    assert math.isfinite(sr) and all(t > 0 for t in tin)
    assert abs(sr - sum(tin)) < 1e-6                               # sum-rate == Σ per-pair
    assert sr > base + 1e-3                                        # precoding beats scalar-AF


def test_multipair_power_budgets_hold():
    """At the optimizer, each Tx and each relay respect their power budgets."""
    scene, active = _scene()
    sr, F, W, tin = optimize_multipair(scene, 2, active, iters=120, restarts=1)
    for f in F:
        assert float((f.abs() ** 2).sum()) <= mp.P_TX + 1e-5
    Ls = _relay_chols_mp(scene, 2, active, F, mp.SIGMA_R)
    for j in range(len(active)):
        Kll = Ls[j] @ Ls[j].mH
        relpow = float(torch.trace(W[j] @ Kll @ W[j].mH).real)
        assert relpow <= mp.P_RELAY + 1e-3


def test_multipair_minrate_beats_scalar_af():
    """Max-min: optimized min-rate exceeds the scalar-AF baseline's min-rate."""
    scene, active = _scene()
    base_min = min(mp.pair_rates(scene, 2, active)["tin"])
    mr, F, W, tin = optimize_multipair(scene, 2, active, objective="minrate",
                                       iters=150, restarts=2)
    assert abs(mr - min(tin)) < 1e-6
    assert mr > base_min + 1e-3


def test_multipair_multistart_at_least_single():
    """Multi-start (restarts>1) is at least the single (seed-0) start."""
    scene, active = _scene()
    one = optimize_multipair(scene, 2, active, iters=120, restarts=1)[0]
    multi = optimize_multipair(scene, 2, active, iters=120, restarts=3)[0]
    assert multi >= one - 1e-6
