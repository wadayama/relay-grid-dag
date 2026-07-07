# tests

Regression tests for `relay-grid-dag` (34 tests across 8 files). Run with:

```bash
uv run pytest
```

Many tests double as the **tie-point tests** `T1`–`T8` of `SPEC.md` §8 — the
paper↔implementation checks named in the manuscript. All are seeded and
deterministic.

## Coverage

| File | Checks |
| --- | --- |
| `test_engine.py` | The precoding engine (`SPEC.md` §4): **T2** AD gradient vs central finite differences, **T4** optimizer ≥ scalar-AF, **T5** power budgets hold at the optimizer, **T6** direct-only `f(∅)` = water-filling, and multi-start ≥ single start. |
| `test_precoded_merge.py` | The precoded merge channel is a faithful superset of the scalar-AF model: defaults byte-identical to the old isotropic path, `F=√(P/n)I` reproduces isotropic, `W=gI` reproduces the scalar-AF relay, and the relay-matrix MI stays differentiable. |
| `test_select_capacity.py` | Selection on the optimized MI: **T7** monotonicity / free disposal, greedy near-oracle and above the received-power baseline, and warm-start ≈ cold-start. |
| `test_viz.py` | **T1** the field intensity at the Rx array equals the model received power `tr(G_eff G_eff^H)` for the same precoders (incl. the no-relay case); carrier-field dB shape/peak. |
| `test_multipair.py` | Multi-pair scalar-AF rates (`cmi-dag`): TIN vs genie-free interference cost, the no-interference limit, shared relays injecting interference, and multi-pair selection value. |
| `test_multipair_precoding.py` | The matrix-relay engine: `optimize_multipair` beats scalar-AF on sum-rate and max-min, power budgets hold, multi-start ≥ single. |
| `test_smoke.py` | End-to-end smoke over the public API: physics gradient, selection value, **near-field rank-one far-field limit** (T3), continuous-to-discrete recovery, wideband per-subcarrier, naive baselines / `K=0`, and `P_relay` plumbing. |
| `test_smoke_precoded.py` | Full-scale precoded end-to-end, wideband per-subcarrier, edge cases, and field-scales-with-precoder. |

The figure-producing studies live in [`../examples/`](../examples/). Two assertions
compare two independent non-convex optimizations within a small tolerance
(`test_warm_start_matches_cold_start`, and the `≥ 0.95 × oracle` bounds); they pass
deterministically but are the most platform-tolerance-sensitive.
