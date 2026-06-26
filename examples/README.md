# examples

Runnable NF-PRG studies and the interactive demo, ported from the original smoke
test. They build on the public `relay_grid_dag` API; `_common.py` is the
examples-layer shim (re-exports the library + matplotlib helpers + an `out/` dir).

```bash
uv sync --extra examples
uv run python examples/<script>.py        # figures -> examples/out/
```

## Studies (figure + PASS/FAIL printout)

| Script | Question |
| --- | --- |
| `s1_position_matters.py` | Does relay position modulate MI? (near vs far) |
| `s2_selection_value.py` | MI-aware selection vs received-power / random |
| `s3_cont_to_discrete.py` | continuous PGA → grid rounding → 1-swap ≈ oracle |
| `s4_ofdm_shared.py` | OFDM shared-activation (beam-squint) cost |
| `s2b_robustness.py` | is the selection value systematic? (60 random configs) |
| `s3b_scaling.py` | scaling L=25→400: greedy cheap & near-optimal |
| `run_all.py` | run S1–S4 and print a PASS/FAIL summary |

## Interactive demo

`relay_grid_demo.py` — drag Tx or Rx; greedy selection of K relays updates in real
time over the candidate grid, with a near-field carrier-field background and a
per-subcarrier rate inset (wideband sum-rate selection).

```bash
uv run python examples/relay_grid_demo.py            # live window (drag, K buttons, q to quit)
uv run python examples/relay_grid_demo.py --snapshot # shareable still -> out/relay_grid_demo.png
uv run python examples/relay_grid_demo.py --selftest # headless check
```

The live window uses matplotlib's default interactive backend for your platform
(macosx / Qt / Tk), so it works cross-platform on any desktop with a GUI toolkit.
On a headless machine (no display), use `--snapshot` / `--selftest`, which render
with the Agg backend and work everywhere.

Note: MI / SE is reported as the idealized full-duplex value `SE (bits/s/Hz, FD)`;
a half-duplex realisation (sum-combining receiver) achieves `SE_HD = (1/2) SE_FD`.
