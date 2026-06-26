# relay-grid-dag

**Near-field Programmable Relay Grid (NF-PRG):** sparse activation / selection of
candidate relays for geometry-aware mutual-information channel shaping.

Given a transmitter, a receiver, and a grid of `L` candidate relay sites, NF-PRG
*activates a few* (`K ≪ L`) to actively shape the near-field channel. This library
provides the **selection / activation layer** — candidate grids, greedy / swap /
exhaustive / received-power / distance strategies, and the continuous-to-discrete
position-gradient pipeline — on top of an analytic near-field physics engine.

```
geometry  ──►  {H_k}  ──►  K-recursion MI  ──►  select K of L relays
 Tx/Rx/grid   spherical    (gaussian-dag)        greedy / swap / oracle
              -wave LoS                          received-power / distance
```

## Architecture and dependencies

This package depends **only on public information**, so it is publishable as-is:

- **`gaussian-dag`** (public, pinned): the K-recursion log-det MI kernel — an
  external dependency, not vendored.
- **near-field physics** (`relay_grid_dag/_nearfield/`): a **vendored, frozen copy**
  of [`nearfield-dag`](https://github.com/wadayama/nearfield-dag) (spherical-wave
  LoS channels + linear-Gaussian DAG specs, MIT, same author). It is kept
  **canonical here** and maintained with this library; the upstream `nearfield-dag`
  repo continues independently as a movable-relay (drone) research theme. Vendoring
  decouples this library's reproducibility from that diverging upstream.

## Install

```bash
git clone https://github.com/wadayama/relay-grid-dag.git
cd relay-grid-dag
uv sync --extra examples     # runtime + matplotlib (examples) + pytest (dev)
uv run pytest
```

`gaussian-dag` is fetched from its public Git repository (pinned in `uv.lock`).

## Quick start

```python
import relay_grid_dag as rgd

scene, names, coords = rgd.build_candidate_scene("near")   # Tx, Rx, 25 candidates
print(rgd.direct_only_mi(scene))                           # direct-only link, bits

# Activate the 3 best relays by greedy MI vs the naive received-power baseline:
g  = rgd.select_greedy_mi(scene, names, 3)
rp = rgd.select_received_power(scene, names, 3)
print(rgd.subset_mi(scene, [names[i] for i in g]))         # smart selection
print(rgd.subset_mi(scene, [names[i] for i in rp]))        # naive selection
```

## Layout

```
relay-grid-dag/
├── relay_grid_dag/
│   ├── _nearfield/      vendored near-field physics (channels, Scene, mi, OFDM, viz)
│   ├── grid.py          candidate-grid geometry + scene builder
│   ├── selection.py     K-of-L selection: greedy / swap / exhaustive / baselines
│   │                    + continuous-to-discrete (position-gradient PGA → rounding)
│   └── __init__.py      public API (physics re-exports + selection layer)
├── experiments/         paper experiments (reproducible figures)   [to be added]
├── tests/               regression tests (physics + selection)
└── pyproject.toml
```

## Roadmap

- Half-duplex two-phase AF model (the paper's primary duplexing assumption: 1/2
  pre-log, `Y1` direct + `Y2` relay-forward) — `halfduplex.py`.
- Multi-pair interference (shared-subcarrier) via the sibling `cmi-dag`.
- Migrate the smoke study (S1–S4, robustness, scaling) into `experiments/`.

## License

MIT — see [`LICENSE`](LICENSE). The vendored `_nearfield/` retains the same MIT
license and authorship.

## Citation

> T. Wadayama and Na Siqi, *Mutual Information Optimization via K-Recursion and
> Automatic Differentiation for Linear Gaussian Wireless Networks*,
> arXiv:2606.06982 [cs.IT], 2026.
