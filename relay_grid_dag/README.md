# relay_grid_dag — library source

The Near-field Relay Grid: a **differentiable joint-precoding engine** (Tx precoder
`F` and per-relay matrices `{W_l}`, optimized by complex-AD projected gradient
ascent) with a **discrete relay-selection layer** on top of the optimized MI, plus a
**multi-pair interference** extension. The physics (channels, K-recursion MI) is
delegated; this package is the precoding, selection, and multi-pair logic on top.

All public symbols are re-exported from the top-level package — see the
[Public API table](../README.md#public-api) and the math in [`../MATH.md`](../MATH.md).

## Modules

| Module | Contents |
| --- | --- |
| `_nearfield/` | **Vendored near-field physics** — a frozen copy of `nearfield-dag` (spherical-wave LoS channels, `Scene`, `mi`, `multirelay_merge`, OFDM, projections, `viz`). Imports rewritten to package-relative; MI still delegates to the public `gaussian-dag` K-recursion. Not modified beyond vendoring. |
| `grid.py` | Candidate-grid geometry: `grid_coords` (row-major `(y,x)` centres) and `build_candidate_scene` (Tx + Rx + `L` candidate relay nodes). Holds the default scene constants (`N_TX`, `N_RX`, `N_RELAY`, `DIRECT_ATTEN`, `P_RELAY`, region). |
| `precoding.py` | The differentiable joint Tx/relay precoding engine (SPEC.md Sec. 4): `optimize_precoders` (whitened complex-AD projected gradient ascent) and `optimized_mi` (the optimized-MI objective `f(S)`). |
| `selection.py` | The unified selection layer on the optimized MI `f(S)`: `subset_mi` / `direct_only_mi` (the conventional scalar-AF relay baseline), `received_power_scores` and `select_received_power` / `select_distance` (naive baselines), `select_greedy`, `select_exhaustive`, `swap_search` (score by `f(S)`; `iters` sets the optimization depth; random-initialized multi-start via `restarts`), `random_subset_stats`, and the continuous-to-discrete pair `continuous_relays` (position-gradient PGA) + `round_to_grid`. |
| `multipair.py` | Multi-pair **scalar-AF** rates via the `cmi-dag` conditional MI: `build_pair_scene`, `pair_rates` (per-pair TIN + genie-free bounds), the sum-rate / max-min objectives, and greedy / exhaustive / received-power selection over `M` pairs. |
| `multipair_precoding.py` | Multi-pair **matrix-relay** engine: `optimize_multipair` (optimize `{F_u}, {W_l}` for the TIN sum-rate or max-min over the multi-root DAG) and `multipair_tin` (the differentiable per-pair TIN objective). |
| `__init__.py` | Public API: re-exports the vendored physics, the precoding engine, the selection layer, and the multi-pair layers. |

## Design boundary

The single-pair engine couples to the physics through `mi(multirelay_merge(...))`
(single-root K-recursion); the multi-pair engine couples through the `cmi-dag`
multi-root conditional MI. The selection strategies are plain combinatorics over
candidate index sets. matplotlib is **not** a core dependency — plotting lives in
`examples/` (the `viz` diagnostics are in the matplotlib tier and imported
explicitly).
