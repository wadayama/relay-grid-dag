# examples

Runnable studies of the **precoded** model: the differentiable joint Tx/relay
precoding engine (SPEC.md Sec. 4) and the selection layer on the optimized
MI `f(S)` (Sec. 5). They build on the public `relay_grid_dag` API;
`_common.py` is the examples-layer shim (re-exports the library + matplotlib
helpers + the canonical paper scenes + an `out/` dir).

```bash
uv sync --extra examples
uv run python examples/<script>.py        # full (paper) depth, figures -> examples/out/
uv run python examples/<script>.py --selftest   # reduced depth, headless check
uv run python examples/run_all.py         # all EXP0-EXP7 at selftest depth
```

## Paper experiment suite (Sec. VII; one script per figure)

Every single-pair experiment uses `_common.canonical_scene()` (the SPEC.md
Sec. 10 frozen scene: 5x5 = 25-candidate grid, 8/8/4-element ULAs); every
two-pair experiment uses `_common.pair_scene()` (the 2-pair interactive demo's
scene: same 5x5 grid, 4/4/4, pairs at (0,±2.5) → (20,±2.5)).

| Script | Question |
| --- | --- |
| `exp0_setup.py` | The deployment, to scale, rendered from the actual Scene objects (setup figure) |
| `exp1_engine.py` | Does PGA converge from random starts, and what does one complex-AD sweep cost vs finite differences? |
| `exp2_precoding_gain.py` | How much does joint Tx/relay precoding buy over the scalar-AF relay, and which lever (Tx vs relay) matters? |
| `exp3_selection.py` | Is greedy(+swap) selection near the exhaustive oracle, and how far above the naive baselines / random subsets? |
| `exp4_nearfield.py` | Where does the gain live: near- vs far-field across the Rayleigh distance |
| `exp5_handover.py` | Mobility: does re-selecting the active set (relay handover) hold the SE as the Rx moves? |
| `exp6_multipair.py` | Capstone: do matrix relays manage two-pair interference where scalar-AF cannot (vs TDMA)? |
| `exp7_mimap.py` | Capstone figure: carrier field vs exact per-stream MI maps — power is not rate |
| `run_all.py` | run EXP0–EXP7 at selftest depth, PASS/FAIL summary |

## Interactive demos

- `relay_grid_demo.py` — single-pair demo (drag the Tx; field / MI display).
- `relay_grid_demo_2pair.py` — two-pair demo (drag any terminal; interference).

## Notes

- **One model, `iters` knob.** The selectors score by the optimized MI
  `f(S)`; `iters` sets the precoder-optimization depth (random-initialized,
  seeded). Each `f(S)` is an inner optimization, so the exhaustive oracle is
  run only where feasible (EXP3 uses K=2: C(25,2)=300 subsets).
- **Consistency.** Visualizations (EXP0 geometry, EXP7 maps) are rendered from
  the same Scene / optimized precoders as the reported numbers (SPEC.md Sec. 6).
- **Duplexing.** MI / SE is the idealized full-duplex value `SE_FD = I(X;Y)`; a
  half-duplex realisation carries a 1/2 prefactor on its two-slot MI
  `SE_HD = (1/2) I(X; Y_1, Y_2)` (paper Remark 2).
- **Archived.** The pre-redesign experiment scripts (`e1`–`e4`, `x_*`,
  `multipair_interference.py`) are in `../archive/examples_v1/`; the earlier
  isotropic studies in `../archive/examples_isotropic/`.
