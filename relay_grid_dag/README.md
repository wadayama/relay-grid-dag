# relay_grid_dag — library source

The Near-field Programmable Relay Grid (NF-PRG) **selection layer**: a fixed
candidate grid of `L` relays, from which a few (`K`) are activated to shape the
near-field channel. The physics (channels, K-recursion MI) is delegated; this
package is the discrete activation / selection logic on top.

All public symbols are re-exported from the top-level package — see the
[Public API table](../README.md#public-api) and the math in [`../MATH.md`](../MATH.md).

## Modules

| Module | Contents |
| --- | --- |
| `_nearfield/` | **Vendored near-field physics** — a frozen copy of `nearfield-dag` (spherical-wave LoS channels, `Scene`, `mi`, `multirelay_merge`, OFDM, projections, `viz`). Imports rewritten to package-relative; MI still delegates to the public `gaussian-dag` K-recursion. Not modified beyond vendoring. |
| `grid.py` | Candidate-grid geometry: `grid_coords` (row-major `(y,x)` centres) and `build_candidate_scene` (Tx + Rx + `L` candidate relay nodes). Holds the default scene constants (`N_TX`, `N_RX`, `N_RELAY`, `DIRECT_ATTEN`, `P_RELAY`, region). |
| `selection.py` | The selection strategies: `subset_mi` / `direct_only_mi` (the single physics call), `received_power_scores` and `select_received_power` / `select_distance` (naive baselines), `select_greedy_mi`, `select_exhaustive`, `swap_search`, `random_subset_stats`, and the continuous-to-discrete pair `continuous_relays` (position-gradient PGA) + `round_to_grid`. |
| `__init__.py` | Public API: re-exports the vendored physics and the selection layer. |

## Design boundary

The single coupling to the physics is the scalar `mi(multirelay_merge(...))`; the
selection strategies are plain combinatorics over candidate index sets. matplotlib
is **not** a core dependency — plotting lives in `examples/` (the `viz` diagnostics
are in the matplotlib tier and imported explicitly).
