# examples

Runnable studies of the **precoded** model: the differentiable joint Tx/relay
precoding engine (SPEC.md Sec. 4) and the selection layer on the optimized
MI `f(S)` (Sec. 5). They build on the public `relay_grid_dag` API;
`_common.py` is the examples-layer shim (re-exports the library + matplotlib
helpers + an `out/` dir).

```bash
uv sync --extra examples
uv run python examples/<script>.py        # figures -> examples/out/
uv run python examples/run_all.py         # run E1-E4, PASS/FAIL summary (~20 s)
```

## Studies (figure + PASS/FAIL printout)

| Script | Question |
| --- | --- |
| `e1_precoding_gain.py` | How much does optimizing the Tx/relay precoders raise the MI over the scalar-AF baseline? |
| `e2_selection_value.py` | Selection on `f(S)`: greedy ≈ oracle (swap polish recovers it), both beat received-power (small grid so the oracle is feasible) |
| `e3_near_vs_far.py` | Why near-field: direct-channel rank and optimized MI, near vs far |
| `e4_carrier_field.py` | The carrier-field map, computed from the *same* optimized precoders as the MI (field intensity at Rx = received signal power) |
| `multipair_interference.py` | 2 interfering pairs: interference-aware selection (cmi-dag CMI) — the separate multi-pair layer (future direction) |
| `run_all.py` | run E1–E4 and print a PASS/FAIL summary |

## Notes

- **One model, `iters` knob.** The selectors score by the optimized MI
  `f(S)`; `iters` sets the precoder-optimization depth (random-initialized). The scalar-AF
  / isotropic baseline (`subset_mi`). Each `f(S)` is an inner optimization, so the
  exhaustive oracle is only run on small grids (E2 uses 6 candidates).
- **Carrier-field visualization.** `viz.carrier_field` takes the optimized
  `(F, {W_l})` and renders the radiated field from those precoders (SPEC.md
  Sec. 6), so the field and the reported MI describe one transmit
  configuration; the field intensity at the Rx array equals the model received
  signal power. E4 produces the field-map figure.
- **Duplexing.** MI / SE is the idealized full-duplex value `SE_FD = I(X;Y)`; a
  half-duplex realisation achieves `SE_HD = (1/2) SE_FD`.
- **Archived.** The earlier isotropic studies (`s1`–`s4`, `s2b`, `s3b`) and an
  earlier interactive demo are in `../archive/examples_isotropic/`.
