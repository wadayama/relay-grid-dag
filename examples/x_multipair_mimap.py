"""Experiment figure (capstone): field vs MI maps for two interfering pairs.

Reuses the 2-pair demo's model (so the figure is the same model as the demo) to render,
at one 2-pair configuration, three panels from the SAME optimized precoders:
  * total carrier field [dB]  -- where the signal *power* goes;
  * I(X0;Y) single-antenna MI map [bits] -- where pair 0's *rate* concentrates;
  * I(X1;Y) MI map [bits].
The MI maps focus on each pair's own Rx and collapse (interference null) at the other's
-- structure the power map cannot show ("power is not rate").

    uv run --extra examples python examples/x_multipair_mimap.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
sys.argv = ["x"]
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from _common import paper_style
paper_style()

import relay_grid_demo_2pair as D

# cleaner-than-interactive settings for a publication figure
D.SEL_ITERS = 20
D.DISP_ITERS = 250
D.DISP_RESTARTS = 3


def run():
    pos = dict(D.INIT_POS)
    s, names = D.build_scene(pos)
    idx = D.select_set(s, names, 2, [True, True], "sumrate", D.P_RELAY)
    rates, intens, F, W, active = D.render(s, names, idx, [True, True], "sumrate", D.P_RELAY)
    print(f"set={idx}  rates={ {u: round(v,2) for u,v in rates.items()} }  "
          f"sum={sum(rates.values()):.2f}")

    # field (cheap) + EXACT per-pixel array MI maps (K-recursion; ~10 s each)
    panels = [D.display_image("field", intens),
              D.exact_mi_map(s, active, F, W, [True, True], 0, pos["rx0"]),
              D.exact_mi_map(s, active, F, W, [True, True], 1, pos["rx1"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (arr, vmn, vmx, lab) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower", extent=[*D.PLANE_X, *D.PLANE_Y], aspect="auto",
                       cmap=D.CMAP, vmin=vmn, vmax=vmx, zorder=0, interpolation="bilinear")
        D._draw(ax, pos, idx, [True, True], ms=10)
        for u in range(D.M):
            for i in idx:
                ax.plot([pos[f"tx{u}"][0], D._COORDS[i, 0], pos[f"rx{u}"][0]],
                        [pos[f"tx{u}"][1], D._COORDS[i, 1], pos[f"rx{u}"][1]],
                        color=D.PAIR_COL[u], ls="--", lw=1.1, alpha=0.7, zorder=5)
        ax.set_xlim(*D.PLANE_X); ax.set_ylim(*D.PLANE_Y); ax.set_title(lab, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Two interfering pairs: carrier field vs MI maps (set {idx}, "
                 f"sum-rate {sum(rates.values()):.1f} bits/s/Hz) -- power is not rate")
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "x_multipair_mimap.pdf"), dpi=120)
    print(f"saved {out}/x_multipair_mimap.pdf")
    return all(v > 0 for v in rates.values())


if __name__ == "__main__":
    print("x_multipair_mimap: field vs MI maps (2 pairs) ...")
    print("PASS" if run() else "FAIL")
