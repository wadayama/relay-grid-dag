# tests

Regression tests for `relay-grid-dag`. Run with:

```bash
uv run pytest
```

## What is covered (`test_smoke.py`)

| Test | Checks |
| --- | --- |
| `test_physics_diamond_mi_and_gradient` | The vendored near-field engine works through the gaussian-dag boundary: the diamond MI is finite and positive, and `I.backward()` gives a native position gradient. |
| `test_selection_value` | MI-aware selection beats the received-power baseline by a margin; greedy is within a few percent of the exhaustive oracle; `K=0` reduces to the direct link. |
| `test_near_field_higher_rank_than_far` | Near-field single-relay MI exceeds the (rank-deficient) far-field link. |
| `test_continuous_to_discrete_recovers_selection` | continuous PGA → grid rounding → 1-swap lands at a strong on-grid set (near the oracle). |

These encode the smoke-study conclusions as fast regressions over the public API, so
the vendored physics and the selection layer stay protected under this library's
ownership. The figure-producing studies live in [`../examples/`](../examples/).
