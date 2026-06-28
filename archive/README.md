# archive/ — superseded code (reference only, not imported)

Snapshots of code replaced by the consolidation onto the single **precoded**
model (SPEC.md). Kept for reference; **not** part of the package and not imported.

- `selection_isotropic.py` — the original isotropic / scalar-AF selection layer.
  Its MI-aware selectors (`select_greedy_mi`, `select_exhaustive`, `swap_search`
  scored by the scalar-AF `subset_mi`) are superseded by the unified selectors in
  `relay_grid_dag/selection.py` (`select_greedy` / `select_exhaustive` /
  `swap_search`), which score by the optimized capacity `f(S)` and recover the old
  isotropic behavior exactly at `iters=0` (since `subset_mi == capacity(iters=0)`).
  The baselines (`subset_mi`, `received_power`, `distance`), `random_subset_stats`,
  and the continuous-to-discrete bridge were carried forward unchanged.

- `examples_isotropic/` — the original example scripts before they were ported to
  the unified API. Git history holds the authoritative versions; these are a
  convenience snapshot.
