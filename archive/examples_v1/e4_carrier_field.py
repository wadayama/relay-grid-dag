"""E4 -- carrier-field map: the radiated signal field of the optimized precoders.

Selects relays on the optimized MI, optimizes the Tx and relay precoders
(F, {W_l}), and renders the carrier (signal) field over space from those same
precoders (SPEC.md Sec. 6). Because the field uses the precoders that also produce
the reported MI, the field and the rate describe one transmit configuration;
the field intensity summed over the Rx array equals the model received signal power
(tie-point T1).
"""
import numpy as np
import torch

import _common as C


def main():
    plt = C._mpl()
    s, names, coords = C.build_candidate_scene("near")
    g = C.select_greedy(s, names, 3, iters=40)              # pick relays on MI
    subset = [names[i] for i in g]
    cap, F, Ws = C.optimize_precoders(s, "tx", subset, "rx", iters=200)
    base = C.subset_mi(s, subset)
    print(f"selected {g}: scalar-AF={base:.2f}  optimized MI={cap:.2f} bits")

    xs = np.linspace(-3.0, 24.0, 170); ys = np.linspace(-9.0, 9.0, 120)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=C.nfd.RDTYPE)
    db = C.nfd.viz.carrier_field(s, "tx", grid, F=F, relays=subset,
                                 relay_mats=Ws).reshape(YY.shape)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    im = ax.imshow(db, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
                   aspect="auto", cmap="magma", vmin=-40, vmax=0)
    C.mark_selected(ax, coords, g, color="cyan", label="active relays")
    ax.scatter(*C.TX_XY, marker="s", c="w", s=80, edgecolors="k", zorder=6, label="Tx")
    ax.scatter(*C.RX_XY, marker="D", c="w", s=80, edgecolors="k", zorder=6, label="Rx")
    ax.set_xlabel("x [wavelengths]"); ax.set_ylabel("y [wavelengths]")
    fig.colorbar(im, ax=ax, label="carrier field [dB]")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    p = f"{C.OUT}/e4_carrier_field.pdf"; fig.savefig(p); print("saved", p)
    return bool(np.isfinite(db).all()) and cap >= base - 1e-6


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
