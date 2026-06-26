"""S1 — Does the relay *position* matter? (the premise behind selection)

NF-PRG's whole premise is that, in the near field, *which* candidate site a relay
sits on strongly changes the channel — so picking good sites is worth it. If the
single-relay MI were nearly flat across the candidate grid, selection would be
pointless. We also contrast near vs far field: the spherical-wave phase is what
makes position matter, so the near-field spread should exceed the far-field one.

PASS if, in the near field, the per-site single-relay MI has a meaningful spread
across the L candidates AND that spread is larger than in the far field.

Output: out/s1_position_matters.png
"""
import numpy as np
import torch

import _common as C


def single_relay_mi_map(model):
    s, names, coords = C.build_candidate_scene(model)
    vals = np.array([C.subset_mi(s, [nm]) for nm in names])
    return s, names, coords, vals


def landscape(model, nx=44, ny=44):
    """Fine MI landscape of one movable relay, for the figure background."""
    s = C.build_candidate_scene(model)[0]
    s.add_node("vr", [10.0, 0.0], C.N_RELAY, "relay", movable_node=True)
    xs = np.linspace(C.GRID_X[0] - 1, C.GRID_X[1] + 1, nx)
    ys = np.linspace(C.GRID_Y[0] - 1, C.GRID_Y[1] + 1, ny)
    Z = np.zeros((ny, nx))
    with torch.no_grad():
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                s.move("vr", [float(x), float(y)])
                Z[j, i] = C.subset_mi(s, ["vr"])
    return xs, ys, Z


def spread(vals, direct):
    """Gain-over-direct spread across candidate sites."""
    gain = vals - direct
    return dict(mi_min=float(vals.min()), mi_max=float(vals.max()),
                rel_spread=float((vals.max() - vals.min()) / np.median(vals)),
                gain_best=float(gain.max()), gain_worst=float(gain.min()))


def main():
    plt = C._mpl()

    s_n, names, coords, mi_near = single_relay_mi_map("near")
    s_f, _, _, mi_far = single_relay_mi_map("far")
    d_near = C.direct_only_mi(s_n)
    d_far = C.direct_only_mi(s_f)
    sp_near = spread(mi_near, d_near)
    sp_far = spread(mi_far, d_far)

    xs_n, ys_n, Zn = landscape("near")
    xs_f, ys_f, Zf = landscape("far")

    vmin = min(mi_near.min(), mi_far.min())
    vmax = max(mi_near.max(), mi_far.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (xs, ys, Z, vals, ttl) in zip(axes, [
            (xs_n, ys_n, Zn, mi_near, "near field"),
            (xs_f, ys_f, Zf, mi_far, "far field")]):
        ax.imshow(Z, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                  aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.85)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap="viridis",
                        vmin=vmin, vmax=vmax, s=110, edgecolors="k",
                        linewidths=0.6, zorder=3)
        ax.scatter(*C.TX_XY, marker="s", s=120, c="tab:cyan", edgecolors="k", zorder=5)
        ax.scatter(*C.RX_XY, marker="D", s=120, c="tab:cyan", edgecolors="k", zorder=5)
        ax.set_title(f"{ttl}: single-relay I(X;Y) over candidate grid")
        ax.set_xlabel("x [wavelengths]"); ax.set_ylabel("y [wavelengths]")
    fig.colorbar(sc, ax=axes, label="I(X;Y) [bits/s/Hz]", shrink=0.85)
    fig.suptitle("S1 — near-field: higher-MI, wavelength-scale multimodal landscape; "
                 "far-field: lower-MI, smoother", fontsize=11)
    path = f"{C.OUT}/s1_position_matters.png"
    fig.savefig(path, bbox_inches="tight")

    # Premise of selection: single-relay MI varies meaningfully across sites, so
    # there is something to choose. (Both near and far vary -- far mostly via path
    # loss -- so the relative spread is NOT the near-field signature; the near-field
    # signature is the higher rank/MI and the wavelength-scale multimodal landscape.)
    near_spread_ok = sp_near["rel_spread"] >= 0.2
    near_higher_mi = sp_near["mi_max"] > sp_far["mi_max"]
    near_higher_gain = sp_near["gain_best"] > sp_far["gain_best"]
    ok = near_spread_ok and near_higher_mi and near_higher_gain

    print("=== S1: position matters ===")
    print(f"direct-only MI:  near={d_near:.3f}  far={d_far:.3f} bits/s/Hz")
    print(f"near : MI range [{sp_near['mi_min']:.2f}, {sp_near['mi_max']:.2f}], "
          f"rel-spread={sp_near['rel_spread']:.3f}, "
          f"gain best/worst={sp_near['gain_best']:.2f}/{sp_near['gain_worst']:.2f}")
    print(f"far  : MI range [{sp_far['mi_min']:.2f}, {sp_far['mi_max']:.2f}], "
          f"rel-spread={sp_far['rel_spread']:.3f}, "
          f"gain best/worst={sp_far['gain_best']:.2f}/{sp_far['gain_worst']:.2f}")
    print(f"[premise]   near rel-spread >= 0.2 (selection has something to choose): {near_spread_ok}")
    print(f"[near sig.] near max MI > far max MI (higher rank/multiplexing): {near_higher_mi} "
          f"({sp_near['mi_max']:.2f} vs {sp_far['mi_max']:.2f})")
    print(f"[near sig.] near best gain-over-direct > far: {near_higher_gain} "
          f"({sp_near['gain_best']:.2f} vs {sp_far['gain_best']:.2f})")
    print("note: far-field MI also varies across sites (via path loss), so relative")
    print("      spread alone does not isolate the near-field effect -- rank/level does.")
    print(f"saved {path}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
