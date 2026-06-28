# relay-grid-dag

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](https://www.python.org/)

**Near-field Relay Grid (NRG):** sparse activation / selection of
candidate relays for geometry-aware mutual-information channel shaping.

Given a transmitter, a receiver, and a grid of `L` candidate relay sites, NRG
*activates a few* (`K ≪ L`) to actively shape the near-field channel. This library
provides the **selection / activation layer** — candidate grids, greedy / swap /
exhaustive / received-power / distance strategies, and the continuous-to-discrete
position-gradient pipeline — on top of an analytic near-field physics engine.

```
geometry  ──►  {H_k}  ──►  K-recursion MI  ──►  select K of L relays
 Tx/Rx/grid   spherical    (gaussian-dag)        greedy / swap / oracle
              -wave LoS                          received-power / distance
```

The selection layer is plain combinatorics on top of one physics call,
`mi(multirelay_merge(...))`; the channel model and the K-recursion are never
reimplemented here. The mathematics is summarised in [`MATH.md`](MATH.md).

## Architecture and dependencies

This package depends **only on public information**, so it is publishable as-is:

- **`gaussian-dag`** (public, pinned): the single-root K-recursion log-det MI
  kernel — an external dependency, not vendored.
- **`cmi-dag`** (public, pinned): the multi-root conditional-MI kernel
  `I(A;B|C)` — used by the multi-pair interference layer; external, not vendored.
- **near-field physics** (`relay_grid_dag/_nearfield/`): a **vendored, frozen copy**
  of [`nearfield-dag`](https://github.com/wadayama/nearfield-dag) (spherical-wave
  LoS channels + linear-Gaussian DAG specs, MIT, same author). It is kept
  **canonical here** and maintained with this library; the upstream `nearfield-dag`
  repo continues independently as a movable-relay (drone) research theme. Vendoring
  decouples this library's reproducibility from that diverging upstream.

## Sister libraries

`relay-grid-dag` is the **candidate-relay selection** member of the Gaussian-DAG
family. They share the K-recursion / complex-autograd design and meet at the
channel matrix `H_k`.

| Library | Scope |
| --- | --- |
| [`gaussian-dag`](https://github.com/wadayama/gaussian-dag) | K-recursion log-det MI on linear Gaussian DAGs (the MI kernel; our dependency). |
| [`nearfield-dag`](https://github.com/wadayama/nearfield-dag) | Near-field spherical-wave channels + movable relays / position autodiff (origin of the vendored physics; now a movable-relay theme). |
| [`cmi-dag`](https://github.com/wadayama/cmi-dag) | Multi-root + conditional MI `I(A;B\|C)` on disjoint subsets — the multi-pair interference engine (integrated; see `multipair.py`). |
| **`relay-grid-dag`** (this repo) | **K-of-L candidate-relay selection** for near-field channel shaping. |

## Requirements

- Python ≥ 3.12
- `gaussian-dag`, `cmi-dag`, PyTorch ≥ 2.12, NumPy ≥ 2.0 (installed as dependencies)
- [`uv`](https://docs.astral.sh/uv/) for environment management (recommended)

## Installation

```bash
git clone https://github.com/wadayama/relay-grid-dag.git
cd relay-grid-dag
uv sync --extra examples     # runtime + matplotlib (examples) + pytest (dev)
uv run pytest                # regression tests (physics + selection)
```

`gaussian-dag` is fetched from its public Git repository (pinned in `uv.lock`); a
fresh clone builds the whole environment in one `uv sync`.

## Repository layout

```
relay-grid-dag/
├── relay_grid_dag/
│   ├── _nearfield/      vendored near-field physics (channels, Scene, mi, OFDM, viz)
│   ├── grid.py          candidate-grid geometry + scene builder
│   ├── selection.py     K-of-L selection: greedy / swap / exhaustive / baselines
│   │                    + continuous-to-discrete (position-gradient PGA → rounding)
│   └── __init__.py      public API (physics re-exports + selection layer)
├── examples/            runnable precoded-model studies (E1–E4) + multi-pair demo
├── archive/             superseded isotropic code (reference only, not imported)
├── docs/                a five-part tutorial walkthrough
├── references/          background notes + key literature
├── tests/               regression tests (physics + selection)
├── MATH.md              mathematical foundations (selection layer)
└── pyproject.toml
```

## Documentation

- [`MATH.md`](MATH.md) — mathematical foundations of the selection layer (merge
  channel MI, the selection problem, the strategies, continuous-to-discrete, OFDM
  shared activation, and the FD/HD duplexing convention).
- [`docs/`](docs/) — a five-part tutorial, from a first selection to the duplexing
  convention.
- Per-directory `README.md` files document the package (`relay_grid_dag/`), the
  runnable studies (`examples/`), the tests (`tests/`), and the references.

## Quick start

```python
import relay_grid_dag as rgd

scene, names, coords = rgd.build_candidate_scene("near")   # Tx, Rx, 25 candidates

# Optimize the Tx + relay precoders for a chosen subset (the MI f(S)):
cap, F, W = rgd.optimize_precoders(scene, "tx", [names[6], names[18]], "rx")
print(cap)                                                 # optimized MI, bits

# Select which K relays to activate, scoring by the optimized MI f(S):
g          = rgd.select_greedy(scene, names, 3, iters=50)        # greedy on MI
sw, sw_cap = rgd.swap_search(scene, names, g, 3, iters=50)       # one-swap polish
rp         = rgd.select_received_power(scene, names, 3)          # naive baseline

# The conventional scalar-AF relay (no precoder optimization) is the baseline:
print(rgd.subset_mi(scene, [names[i] for i in g]))              # scalar-AF baseline, bits
best, best_cap = rgd.select_exhaustive(scene, names, 2, iters=50)  # exhaustive on MI (small L)
```

For wideband (OFDM), evaluate per-subcarrier and sum:

```python
k_waves = rgd.subcarrier_wavenumbers(f_c=10.0, S=8, df=0.4)
sumrate = sum(rgd.subset_mi(scene, [names[i] for i in g], k_wave=k) for k in k_waves)
```

See [`examples/`](examples/) for the runnable studies (E1–E4), including the
carrier-field map (`e4_carrier_field.py`).

## Public API

All symbols below are re-exported from the top-level `relay_grid_dag` package.

### Grid and scene (this package)

| Symbol | Purpose |
| --- | --- |
| `grid_coords(nx=5, ny=5, xr=(6,14), yr=(-4,4))` | `(L, 2)` candidate relay centres, row-major in `(y, x)`. |
| `build_candidate_scene(model="near", *, coords=None, n_tx=8, n_rx=8, n_relay=4, direct_atten=0.3, …)` | Scene with Tx, Rx and `L` candidate relays `c0..c{L-1}`. Returns `(scene, names, coords)`. |

### Precoding engine (this package) — the MI oracle `f(S)`

| Symbol | Purpose |
| --- | --- |
| `optimize_precoders(scene, src, subset, dst, *, iters=400, restarts=1, seed=0, P_tx=100, P_relay=100, k_wave=2π, d=None, F_init=None, W_init=None)` | Maximize the merge-channel MI over the Tx precoder `F` and relay matrices `{W_l}` by whitened complex-AD projected gradient ascent → `(mi_bits, F, W_list)`. Random-initialized; `restarts` runs a multi-start and keeps the best. |
| `optimized_mi(scene, src, subset, dst, **kw)` | The optimized MI `f(S)` (bits) only — wrapper over `optimize_precoders`. |

### Baseline MI (this package)

| Symbol | Purpose |
| --- | --- |
| `subset_mi(scene, subset, *, k_wave=2π, P_relay=100)` | Conventional scalar-AF relay MI in bits (the baseline; no precoder optimization). `subset=[]` is direct-only. FD value (see Conventions). |
| `direct_only_mi(scene, *, k_wave=2π)` | Scalar-AF MI of the direct link alone (`subset=[]`). |

### Selection strategies (this package) — on the MI `f(S)`

| Symbol | Purpose |
| --- | --- |
| `select_greedy(scene, names, K, *, warm_start=True, **engine_kw)` | Forward greedy on `f(S)`, warm-started. `engine_kw` (e.g. `iters`, `restarts`) is forwarded to `optimize_precoders`. |
| `select_exhaustive(scene, names, K, **engine_kw)` | Brute-force best `K`-subset on `f(S)` → `(indices, mi)`. Small `L` (each `f(S)` is an optimization). |
| `swap_search(scene, names, init, K, *, max_passes=8, **engine_kw)` | 1-swap local search on `f(S)` → `(indices, mi)`; polishes greedy. |
| `select_received_power(scene, names, K)` | Top-`K` by two-hop cascade gain `‖H_rd H_tr‖²_F` (naive baseline). |
| `select_distance(coords, K, tx, rx)` | `K` shortest two-hop path lengths (naive baseline). |
| `received_power_scores(scene, names)` | Per-candidate received-power scores. |
| `random_subset_stats(scene, names, K, *, trials=200, seed=0)` | mean/std/max/min scalar-AF MI over random `K`-subsets. |

### Continuous-to-discrete (this package)

| Symbol | Purpose |
| --- | --- |
| `continuous_relays(K, lo, hi, *, starts=4, iters=300, d_min=1.8, return_all=False)` | Multi-start position-gradient PGA on `K` virtual relays → `(centres, MI)` (or all starts). |
| `round_to_grid(final, coords)` | Map virtual relay centres to nearest free candidate sites. |

### Multi-pair interference (this package, via `cmi-dag`)

`M` Tx/Rx pairs share the relay grid; each active relay forwards every transmitter
(shared medium), so it is double-edged — it can carry interference too. Rates come
from the multi-root conditional MI.

| Symbol | Purpose |
| --- | --- |
| `build_pair_scene(pairs=…, *, n_relay=4)` | Scene with `M` pairs (`tx{u}`/`rx{u}`) + candidate grid. Returns `(scene, names, coords)`. |
| `pair_rates(scene, M, active, *, cross_atten=0.3, …)` | Per-pair `{"tin": [..], "free": [..]}`: `R_u^TIN = I(X_u;Y_u\|∅)` (interference as noise) and `R_u^free = I(X_u;Y_u\|X_others)` (genie bound). |
| `weighted_sum_rate(scene, M, active, *, weights=None)` | `Σ_u w_u R_u^TIN` objective. |
| `min_rate(scene, M, active)` | Max-min objective `min_u R_u^TIN`. |
| `select_greedy_sumrate(scene, M, names, K, *, objective=…)` | Greedy on the multi-pair objective. |
| `select_exhaustive_sumrate(scene, M, names, K, *, objective=…)` | Brute-force best K-subset (small L). |
| `received_power_pairs(scene, M, names, K)` | Interference-blind baseline (desired cascade gain). |

### Physics (re-exported from the vendored near-field engine)

| Symbol | Purpose |
| --- | --- |
| `Scene` | Named multi-node scene (`source`/`sink`/`relay`); produces channels `H_k`. |
| `mi(spec, *, bits=True)` | MI of a DAG spec via the gaussian-dag K-recursion (bits by default). |
| `multirelay_merge(scene, src, relays, dst, *, P_relay=100, direct=True, k_wave=2π)` | `K` AF relays + optional direct path merging at `Y`. |
| `diamond`, `single_link`, `ris_branch` | 3-/2-node DAG specs. |
| `near_field_channel`, `far_field_channel`, `distances` | Spherical-/planar-wave LoS channels. |
| `subcarrier_wavenumbers(f_c, S, df)`, `subcarrier_frequencies`, `wideband_mi` | OFDM subcarrier grid + wideband sum-rate. |
| `project_box`, `project_min_separation`, `repulsion_penalty` | PGA projectors / penalties. |
| `ula`, `movable`, `ArrayGeometry` | Array geometry. |
| `DTYPE`, `RDTYPE`, `K_WAVE` | `complex128`, `float64`, carrier wavenumber `2π`. |
| `viz` | matplotlib diagnostics (`carrier_field`, `rxpow`, `effrank`) — imported explicitly. |

## Conventions

- **Geometry units.** Lengths are in carrier wavelengths (λ_c = 1, carrier
  wavenumber `k = 2π`); an OFDM subcarrier `i` uses `k_i = 2π f_i/f_c`. Positions are
  2-D `(x, y)`.
- **Candidate indexing.** `grid_coords` is row-major in `(y, x)`: index `r*nx + c`
  is `(x_c, y_r)`. A relay subset is a list of candidate indices (into `names`/`coords`).
- **MI units / duplexing.** `mi()` / `subset_mi()` return **bits/s/Hz** and are the
  **idealized full-duplex** value `SE_FD = I(X;Y)` — see Duplexing convention.
- **Power.** `P_tx` (Tx) and `P_relay` (per-active-relay output) are fixed system
  parameters; `P_relay` is the relay-vs-Tx power knob. AF gain is a deterministic,
  power-normalised scalar, so each node stays an affine-linear map.
- **Near vs far.** `model="near"` is the spherical-wave channel; `model="far"` is the
  planar-wave (rank-deficient) baseline for near-vs-far comparison.

## Duplexing convention

`mi()` / `subset_mi()` return the **idealized full-duplex** value `SE_FD = I(X;Y)`:
the leakage-free model with perfect self-interference cancellation and no
inter-relay coupling = the acyclic linear-Gaussian DAG exactly as computed (the
best-case FD upper bound). A half-duplex realisation with a **sum-combining
receiver** (Rx coherently adds its phase-1 direct and phase-2 relay observations)
achieves `SE_HD = (1/2) SE_FD`. We keep the ½ out of the code: figures/tables report
`SE (bits/s/Hz, FD)` and the paper states `SE_HD = (1/2) SE_FD` explicitly.

## Known limitations

- **Channel model.** Pure line-of-sight spherical-wave (near) / plane-wave (far).
  No multipath, scattering, or blockage beyond a scalar `direct_atten` on the direct
  edge (an abstraction of NLoS, not an explicit obstacle). For ray-traced multipath,
  use the sibling `radio-twin`.
- **Array geometry.** Uniform linear arrays only.
- **Relay model.** Amplify-and-forward with a power-normalised *scalar* gain
  (ensemble statistics), so the linear-Gaussian precondition of the K-recursion holds
  exactly. Instantaneous per-symbol power constraints are out of scope.
- **Duplexing.** Idealized full-duplex (leakage-free DAG); the half-duplex penalty is
  a reporting convention (× ½), not modelled per-slot. True FD with self/inter-relay
  leakage would create cycles that break the acyclic K-recursion.
- **Optimization.** Selection is combinatorial; greedy is near-oracle but not
  optimal, and the continuous PGA is non-convex (multi-start recommended). Exhaustive
  search is feasible only at small `L`.
- **Multi-pair.** Rates use Gaussian inputs and treat interference as noise (TIN);
  `pair_rates` also returns the interference-free genie bound. Joint encoding/
  decoding, rate splitting, and per-subcarrier OFDM multi-pair are out of scope.

## Roadmap

- Multi-pair interference (shared-subcarrier) via the sibling `cmi-dag`.

## Citation

The methods consumed here — the K-recursion, the log-det MI, and the
Wirtinger-gradient PGA — are the version of record in:

> T. Wadayama and Na Siqi, *Mutual Information Optimization via K-Recursion and
> Automatic Differentiation for Linear Gaussian Wireless Networks*,
> arXiv:2606.06982 [cs.IT], 2026. <https://arxiv.org/abs/2606.06982>

## Acknowledgement

This work was supported by JST, CRONOS, Japan Grant Number **JPMJCS25N5**.

## License

MIT — see [`LICENSE`](LICENSE). The vendored `_nearfield/` retains the same MIT
license and authorship.
