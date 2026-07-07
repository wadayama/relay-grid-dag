"""Interactive 2-pair demo for the precoded Near-field Relay Grid.

Two Tx/Rx pairs share one candidate relay grid. Drag any terminal (tx0/rx0 = pair 0,
tx1/rx1 = pair 1). On release, K relays are activated by greedy on the optimized
multi-pair objective (TIN sum-rate or max-min) over the pairs currently switched ON,
and the Tx precoders {F_u} and relay matrices {W_l} are optimized.

Two display modes (switch on the left, no re-optimization):
  * field  -- the total carrier field, the superposed signal energy of the active
              Tx / relays (Tx0 + Tx1 contributions), in dB.
  * MI Xu  -- a single-antenna "MI map": at each point p (a 1-antenna probe Y),
              I(X_u; Y) = log2(1 + ||g_u(p)||^2 / (sigma_d^2 + sum_{v!=u} ||g_v(p)||^2)),
              the TIN spectral efficiency for stream u treating the other as noise.
              Reveals the *interference null* (where I(X_u;.) collapses) directly.
              Illustrative single-antenna probe -- the reported R_u use the Rx array.

Toggle a pair off to compare simultaneous service (interference managed by the matrix
relays) vs serving one alone; simultaneous sum-rate can beat TDMA (half the sum of the
alone-rates) -- a spatial multiplexing gain. Rates are bits/s/Hz, idealized full-duplex.

Controls: drag terminals; radios K / relay power / objective / display; check pairs on-off; q quits.
    uv run --extra examples python examples/relay_grid_demo_2pair.py            # window
    uv run --extra examples python examples/relay_grid_demo_2pair.py --snapshot # -> out/
    uv run --extra examples python examples/relay_grid_demo_2pair.py --selftest # headless
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
SELFTEST = "--selftest" in sys.argv
SNAPSHOT = "--snapshot" in sys.argv
if SELFTEST or SNAPSHOT:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import torch

import relay_grid_dag as rgd
import relay_grid_dag.multipair as mp
from relay_grid_dag import viz
from relay_grid_dag.multipair_precoding import optimize_multipair, multipair_tin

# ------------------------------- configuration -------------------------------
M = 2
DIRECT_ATTEN = 0.3
CROSS_ATTEN = 0.3
P_RELAY = 100.0
SIGMA_D2 = 1.0                  # destination noise power (matches the rate model)
N_TX = N_RX = 4
N_RELAY = 4
PLANE_X = (-3.0, 23.0)
PLANE_Y = (-7.0, 7.0)
GRID_X = (6.0, 14.0)
GRID_Y = (-4.0, 4.0)
GRID_NX = GRID_NY = 5            # L = 25 candidate sites (dynamic selection)
MAX_K = 3
HEAT_NX, HEAT_NY = 130, 95
CMAP = "viridis"
PAIR_COL = ["#ff7a00", "#00d4ff"]     # pair 0 (orange), pair 1 (cyan)
OFF_COL = "0.55"
SELC = "#ff2bd6"                       # activated relays (magenta)
SEL_ITERS = 5                  # greedy-selection depth (selection stable at L=25; snappy)
DISP_ITERS = 80
DISP_RESTARTS = 1              # single start for the shown set (demo: speed over exact rate)

INIT_POS = {"tx0": (0.0, 2.5), "rx0": (20.0, 2.5),
            "tx1": (0.0, -2.5), "rx1": (20.0, -2.5)}

_COORDS = mp.grid_coords(GRID_NX, GRID_NY, GRID_X, GRID_Y)
_gx, _gy = np.meshgrid(np.linspace(*PLANE_X, HEAT_NX), np.linspace(*PLANE_Y, HEAT_NY))
_GRID = torch.tensor(np.stack([_gx.ravel(), _gy.ravel()], axis=1), dtype=rgd.RDTYPE)
_ZERO = np.zeros((HEAT_NY, HEAT_NX))

SPECS = (f"near-field LoS, 2 pairs (TIN)  |  Tx/Rx: {N_TX}-elem ULA, relay: {N_RELAY}-elem ULA "
         f"(0.5λ spacing)  |  {GRID_NX}x{GRID_NY} relay grid, "
         f"{(GRID_X[1]-GRID_X[0])/(GRID_NX-1):.1f}x{(GRID_Y[1]-GRID_Y[0])/(GRID_NY-1):.1f}λ "
         f"(L={GRID_NX*GRID_NY})  |  P_tx=P_relay  |  direct/cross atten {DIRECT_ATTEN}/{CROSS_ATTEN}")


def build_scene(pos):
    pairs = [(pos["tx0"], pos["rx0"]), (pos["tx1"], pos["rx1"])]
    s, names, _ = mp.build_pair_scene(pairs=pairs, coords=_COORDS,
                                      n_tx=N_TX, n_rx=N_RX, n_relay=N_RELAY,
                                      direct_atten=DIRECT_ATTEN)
    return s, names


def _on_pairs(on):
    return [u for u in range(M) if on[u]]


def select_set(scene, names, K, on, objective, P_relay):
    """Greedy activation of K relays on the optimized objective over the ON pairs."""
    pairs = _on_pairs(on)
    if K <= 0 or not pairs:
        return []
    chosen, remaining = [], set(range(len(names)))
    for _ in range(K):
        best_j, best = None, -1e18
        for j in remaining:
            active = [names[i] for i in chosen + [j]]
            val = optimize_multipair(scene, pairs, active, objective=objective, P_relay=P_relay,
                                     direct_atten=DIRECT_ATTEN, cross_atten=CROSS_ATTEN,
                                     iters=SEL_ITERS, restarts=1)[0]
            if val > best:
                best, best_j = val, j
        chosen.append(best_j); remaining.discard(best_j)
    return sorted(chosen)


def render(scene, names, idx, on, objective, P_relay):
    """Optimize precoders for the ON pairs; return
    ``({pair: rate}, {pair: intensity_2d}, F_list, W_list, active)``. Each intensity is
    the linear carrier field ||g_u(p)||^2 over the grid."""
    pairs = _on_pairs(on)
    if not pairs:
        return {}, {}, [], [], []
    active = [names[i] for i in idx]
    _, F, W, tin = optimize_multipair(scene, pairs, active, objective=objective, P_relay=P_relay,
                                      direct_atten=DIRECT_ATTEN, cross_atten=CROSS_ATTEN,
                                      iters=DISP_ITERS, restarts=DISP_RESTARTS)
    rates, intens = {}, {}
    for k, u in enumerate(pairs):
        rates[u] = float(tin[k])
        intens[u] = viz.field_intensity(scene, f"tx{u}", _GRID, F=F[k], relays=active,
                                        relay_mats=W).numpy().reshape(HEAT_NY, HEAT_NX)
    return rates, intens, F, W, active


def display_image(mode, intens):
    """Cheap heatmap (field / single-antenna SINR MI) -> (array, vmin, vmax, label)."""
    if mode == "field":
        if not intens:
            return np.full((HEAT_NY, HEAT_NX), -40.0), -40, 0, "total carrier field [dB]"
        lin = sum(intens.values())
        db = 10.0 * np.log10(np.maximum(lin, 1e-30) / max(lin.max(), 1e-30))
        return np.maximum(db, -40.0), -40, 0, "total carrier field [dB]"
    u = int(mode.split("X")[1])                                    # "MI X0" -> 0
    if u not in intens:
        return _ZERO, 0, 1, f"I(X{u};Y) [bits] (pair {u} off)"
    interf = sum(intens[v] for v in intens if v != u)
    Iu = np.log2(1.0 + intens[u] / (SIGMA_D2 + interf))
    return Iu, 0.0, max(float(Iu.max()), 1e-6), f"MI (SINR): I(X{u};Y) [bits]"


def exact_mi_map(scene, active, F, W, on, pair, home):
    """EXACT per-pixel array MI map I(X_pair;Y) via cmi-dag: place rx{pair}'s array at
    each grid point and evaluate the conditional MI under the fixed precoders (~10 s).
    Captures array combining + spatial interference nulling that the SINR map omits."""
    pairs = _on_pairs(on)
    if pair not in pairs:
        return _ZERO, 0, 1, f"I(X{pair};Y) exact [bits] (pair {pair} off)"
    k = pairs.index(pair)
    vals = np.zeros(len(_GRID))
    for j, p in enumerate(_GRID):
        scene.move(f"rx{pair}", [float(p[0]), float(p[1])])
        vals[j] = float(multipair_tin(scene, pairs, active, F, W,
                                      direct_atten=DIRECT_ATTEN, cross_atten=CROSS_ATTEN)[k])
    scene.move(f"rx{pair}", list(home))
    arr = vals.reshape(HEAT_NY, HEAT_NX)
    return arr, 0.0, max(float(arr.max()), 1e-6), f"MI (exact): I(X{pair};Y) [bits]"


def _info(rates, on, K, pmult):
    db = int(round(10 * math.log10(pmult)))
    def row(lab, val):                                  # left-justified label, aligned '='
        return f"{lab:<12}= {val}"
    lines = [row("K activated", f"{K} of {len(_COORDS)}"),
             row("relay power", f"+{db} dB rel. Tx"), ""]
    ssum = 0.0
    for u in range(M):
        if on[u]:
            r = rates.get(u, 0.0); ssum += r
            lines.append(row(f"R{u} (pair {u})", f"{r:6.2f} bits/s/Hz"))
        else:
            lines.append(row(f"R{u} (pair {u})", "off"))
    lines.append(row("sum-rate", f"{ssum:6.2f} bits/s/Hz"))
    return "\n".join(lines)


def _draw(ax, pos, idx, on, *, ms=14):
    relay_sc = ax.scatter(_COORDS[idx, 0], _COORDS[idx, 1], s=240, facecolors="none",
                          edgecolors=SELC, linewidths=2.4, zorder=7)
    dl, mk, lbl = {}, {}, {}
    for u in range(M):
        c = PAIR_COL[u] if on[u] else OFF_COL
        a = 0.9 if on[u] else 0.35
        (dl[u],) = ax.plot([pos[f"tx{u}"][0], pos[f"rx{u}"][0]],
                           [pos[f"tx{u}"][1], pos[f"rx{u}"][1]], color=c, ls="-",
                           lw=1.4, alpha=a, zorder=4)
        for kind, mar in (("Tx", "^"), ("Rx", "v")):
            t = f"{kind.lower()}{u}"
            (mk[t],) = ax.plot([pos[t][0]], [pos[t][1]], mar, color=c, ms=ms,
                               mec="k", mew=0.6, alpha=a, zorder=8)
            lbl[t] = ax.text(pos[t][0], pos[t][1] + 0.8, f"{kind}{u}", color=c, fontsize=9,
                             fontweight="bold", ha="center", va="bottom", alpha=a, zorder=9,
                             path_effects=[pe.withStroke(linewidth=2, foreground="k")])
    cl = {}                                            # cross (interference) direct links tx_v -> rx_u
    for v in range(M):
        for u in range(M):
            if v == u:
                continue
            vis = on[v] and on[u]
            (cl[(v, u)],) = ax.plot([pos[f"tx{v}"][0], pos[f"rx{u}"][0]],
                                    [pos[f"tx{v}"][1], pos[f"rx{u}"][1]],
                                    color=PAIR_COL[v], ls=":", lw=1.6,
                                    alpha=(0.45 if vis else 0.0), zorder=3.5)
    ax.scatter(_COORDS[:, 0], _COORDS[:, 1], s=55, c="0.8", edgecolors="k",
               linewidths=0.3, zorder=3)
    return relay_sc, dl, mk, lbl, cl


def _set_relay_paths(path_lns, pos, idx, on):
    """Dotted two-hop routes tx_u -> relay -> rx_u, in each pair's colour, for each ON
    pair x active relay."""
    k = 0
    for u in range(M):
        if not on[u]:
            continue
        for i in idx:
            ln = path_lns[k]
            ln.set_data([pos[f"tx{u}"][0], _COORDS[i, 0], pos[f"rx{u}"][0]],
                        [pos[f"tx{u}"][1], _COORDS[i, 1], pos[f"rx{u}"][1]])
            ln.set_color(PAIR_COL[u]); ln.set_alpha(0.7)
            k += 1
    for j in range(k, len(path_lns)):
        path_lns[j].set_data([], [])


# --------------------------------- self-test ---------------------------------
def selftest():
    print("selftest: 2-pair demo (field + MI map) ...")
    pos = dict(INIT_POS)
    s, names = build_scene(pos)
    idx = select_set(s, names, 2, [True, True], "sumrate", P_RELAY)
    rates, intens, *_ = render(s, names, idx, [True, True], "sumrate", P_RELAY)
    fld, *_ = display_image("field", intens)
    mi0, _, mx0, _ = display_image("MI X0", intens)
    mi1, _, mx1, _ = display_image("MI X1", intens)
    assert np.isfinite(fld).all() and np.isfinite(mi0).all() and np.isfinite(mi1).all()
    assert all(v > 0 for v in rates.values()) and mx0 > 0 and mx1 > 0
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (arr, vmn, vmx, lab) in zip(axes, [display_image("field", intens),
                                               display_image("MI X0", intens),
                                               display_image("MI X1", intens)]):
        im = ax.imshow(arr, origin="lower", extent=[*PLANE_X, *PLANE_Y], aspect="auto",
                       cmap=CMAP, vmin=vmn, vmax=vmx, zorder=0)
        _draw(ax, pos, idx, [True, True], ms=10)
        for u in range(M):
            for i in idx:
                ax.plot([pos[f"tx{u}"][0], _COORDS[i, 0], pos[f"rx{u}"][0]],
                        [pos[f"tx{u}"][1], _COORDS[i, 1], pos[f"rx{u}"][1]],
                        color=PAIR_COL[u], ls="--", lw=1.1, alpha=0.7, zorder=5)
        ax.set_xlim(*PLANE_X); ax.set_ylim(*PLANE_Y); ax.set_title(lab, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"2-pair  set={idx}  sum={sum(rates.values()):.2f}  "
                 f"(field | MI X0 | MI X1; MI peaks {mx0:.1f}/{mx1:.1f} bits)")
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "relay_grid_demo_2pair_selftest.png"), dpi=110)
    print(f"  set={idx}  R={[round(v,2) for v in rates.values()]} sum={sum(rates.values()):.2f} "
          f"| MI-map peaks X0={mx0:.2f} X1={mx1:.2f} bits\nselftest passed.")


# --------------------------------- snapshot ----------------------------------
def snapshot():
    pos = dict(INIT_POS)
    s, names = build_scene(pos)
    idx = select_set(s, names, 2, [True, True], "sumrate", P_RELAY)
    rates, intens, *_ = render(s, names, idx, [True, True], "sumrate", P_RELAY)
    arr, vmn, vmx, lab = display_image("MI X0", intens)        # show an MI map by default
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    plt.subplots_adjust(left=0.16)
    im = ax.imshow(arr, origin="lower", extent=[*PLANE_X, *PLANE_Y], aspect="auto",
                   cmap=CMAP, vmin=vmn, vmax=vmx, zorder=0)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label(lab)
    _draw(ax, pos, idx, [True, True])
    for u in range(M):
        for i in idx:
            ax.plot([pos[f"tx{u}"][0], _COORDS[i, 0], pos[f"rx{u}"][0]],
                    [pos[f"tx{u}"][1], _COORDS[i, 1], pos[f"rx{u}"][1]],
                    color=PAIR_COL[u], ls="--", lw=1.1, alpha=0.7, zorder=5)
    ax.set_xlim(*PLANE_X); ax.set_ylim(*PLANE_Y)
    ax.set_xlabel(r"x [$\lambda$]"); ax.set_ylabel(r"y [$\lambda$]")
    ax.text(0.02, 0.98, _info(rates, [True, True], 2, 1), transform=ax.transAxes, va="top",
            fontsize=10, family="monospace", bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_title("2-pair NRG: MI map I(X0;Y) -- note the interference null near rx1")
    fig.text(0.5, 0.012, SPECS, ha="center", va="bottom", fontsize=8, color="0.35")
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "relay_grid_demo_2pair.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}  (set={idx}, sum-rate={sum(rates.values()):.2f})")


# -------------------------------- interactive --------------------------------
def interactive():
    from matplotlib.widgets import RadioButtons, CheckButtons

    st = {"pos": dict(INIT_POS), "drag": None, "K": 2, "pmult": 1, "obj": "sumrate",
          "on": [True, True], "idx": [], "disp": "field", "intens": {}, "rates": {},
          "F": [], "W": [], "active": [], "exact": {}}
    P = lambda: st["pmult"] * P_RELAY
    s, names = build_scene(st["pos"])

    matplotlib.rcParams["toolbar"] = "None"
    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    plt.subplots_adjust(left=0.18)
    ax.set_xlim(*PLANE_X); ax.set_ylim(*PLANE_Y)
    ax.set_xlabel(r"x [$\lambda$]"); ax.set_ylabel(r"y [$\lambda$]")

    st["idx"] = select_set(s, names, st["K"], st["on"], st["obj"], P())
    st["rates"], st["intens"], st["F"], st["W"], st["active"] = \
        render(s, names, st["idx"], st["on"], st["obj"], P())
    arr, vmn, vmx, lab = display_image(st["disp"], st["intens"])
    im = ax.imshow(arr, origin="lower", extent=[*PLANE_X, *PLANE_Y], aspect="auto",
                   cmap=CMAP, vmin=vmn, vmax=vmx, zorder=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046); cbar.set_label(lab)
    relay_sc, dl, mk, lbl, cl = _draw(ax, st["pos"], st["idx"], st["on"], ms=15)
    path_lns = [ax.plot([], [], ls="--", lw=1.1, alpha=0.7, zorder=5)[0]
                for _ in range(M * MAX_K)]
    _set_relay_paths(path_lns, st["pos"], st["idx"], st["on"])
    txt = ax.text(0.02, 0.98, _info(st["rates"], st["on"], st["K"], st["pmult"]),
                  transform=ax.transAxes, va="top", fontsize=10, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_title("2-pair NRG: drag tx/rx (orange=pair0, cyan=pair1); switch field / MI map")
    fig.text(0.5, 0.012, SPECS, ha="center", va="bottom", fontsize=8, color="0.35")

    BX, BW = 0.018, 0.11                       # uniform control-box x / width
    PMULTS = [1, 10, 100]
    plabels = [f"+{int(round(10 * np.log10(m)))} dB" for m in PMULTS]
    ax_k = fig.add_axes([BX, 0.70, BW, 0.18]); ax_k.set_title("K", fontsize=9)
    radio_k = RadioButtons(ax_k, [str(k) for k in range(0, MAX_K + 1)], active=st["K"])
    ax_p = fig.add_axes([BX, 0.55, BW, 0.12]); ax_p.set_title("relay power", fontsize=9)
    radio_p = RadioButtons(ax_p, plabels, active=PMULTS.index(st["pmult"]))
    ax_o = fig.add_axes([BX, 0.44, BW, 0.08]); ax_o.set_title("objective", fontsize=9)
    radio_o = RadioButtons(ax_o, ["sumrate", "minrate"], active=0)
    ax_d = fig.add_axes([BX, 0.22, BW, 0.18]); ax_d.set_title("display", fontsize=9)
    radio_d = RadioButtons(ax_d, ["field", "X0 SINR", "X1 SINR", "X0 exact", "X1 exact"], active=0)
    ax_c = fig.add_axes([BX, 0.10, BW, 0.08]); ax_c.set_title("pairs on/off", fontsize=9)
    check = CheckButtons(ax_c, ["pair 0", "pair 1"], st["on"])
    for _r in (radio_k, radio_p, radio_o, radio_d):
        for _t in _r.labels:
            _t.set_fontsize(8)
    for _t in check.labels:
        _t.set_fontsize(8)

    def redraw():
        mode = st["disp"]
        if mode == "field":
            arr, vmn, vmx, lab = display_image("field", st["intens"])
        else:
            u = int("".join(ch for ch in mode if ch.isdigit()))
            if "exact" in mode:                          # ~10 s per pair, lazy + cached
                if u not in st["exact"]:
                    st["exact"][u] = exact_mi_map(s, st["active"], st["F"], st["W"],
                                                  st["on"], u, st["pos"][f"rx{u}"])
                arr, vmn, vmx, lab = st["exact"][u]
            else:
                arr, vmn, vmx, lab = display_image(f"MI X{u}", st["intens"])
        im.set_data(arr); im.set_clim(vmn, vmx); cbar.set_label(lab)
        fig.canvas.draw_idle()

    def recompute(heavy):
        for u in range(M):
            s.move(f"tx{u}", list(st["pos"][f"tx{u}"]))
            s.move(f"rx{u}", list(st["pos"][f"rx{u}"]))
        if heavy:
            st["idx"] = select_set(s, names, st["K"], st["on"], st["obj"], P())
        idx = st["idx"]
        relay_sc.set_offsets(_COORDS[idx].reshape(-1, 2) if idx else np.empty((0, 2)))
        for u in range(M):
            c = PAIR_COL[u] if st["on"][u] else OFF_COL
            a = 0.9 if st["on"][u] else 0.35
            dl[u].set_data([st["pos"][f"tx{u}"][0], st["pos"][f"rx{u}"][0]],
                           [st["pos"][f"tx{u}"][1], st["pos"][f"rx{u}"][1]])
            dl[u].set_color(c); dl[u].set_alpha(a)
            for t in (f"tx{u}", f"rx{u}"):
                mk[t].set_data([st["pos"][t][0]], [st["pos"][t][1]])
                mk[t].set_color(c); mk[t].set_alpha(a)
                lbl[t].set_position((st["pos"][t][0], st["pos"][t][1] + 0.8))
                lbl[t].set_color(c); lbl[t].set_alpha(a)
        for v in range(M):                               # cross (interference) links
            for u in range(M):
                if v == u:
                    continue
                cl[(v, u)].set_data([st["pos"][f"tx{v}"][0], st["pos"][f"rx{u}"][0]],
                                    [st["pos"][f"tx{v}"][1], st["pos"][f"rx{u}"][1]])
                cl[(v, u)].set_alpha(0.4 if (st["on"][v] and st["on"][u]) else 0.0)
        _set_relay_paths(path_lns, st["pos"], idx, st["on"])
        if heavy:
            st["rates"], st["intens"], st["F"], st["W"], st["active"] = \
                render(s, names, idx, st["on"], st["obj"], P())
            st["exact"] = {}                              # invalidate exact-map cache
            txt.set_text(_info(st["rates"], st["on"], st["K"], st["pmult"]))
            redraw()
        else:
            txt.set_text(f"K activated  = {st['K']} of {len(_COORDS)}\n(release to optimize)")
            fig.canvas.draw_idle()

    def on_press(e):
        if e.inaxes is ax and e.xdata is not None:
            for key in ("tx0", "rx0", "tx1", "rx1"):
                if abs(e.xdata - st["pos"][key][0]) < 1.5 and abs(e.ydata - st["pos"][key][1]) < 1.5:
                    st["drag"] = key
                    break

    def on_motion(e):
        if st["drag"] and e.inaxes is ax and e.xdata is not None:
            st["pos"][st["drag"]] = (float(np.clip(e.xdata, *PLANE_X)),
                                     float(np.clip(e.ydata, *PLANE_Y)))
            recompute(heavy=False)

    def on_release(e):
        if st["drag"]:
            st["drag"] = None
            recompute(heavy=True)

    def on_check(label):
        i = ["pair 0", "pair 1"].index(label)
        st["on"][i] = not st["on"][i]
        recompute(heavy=True)

    radio_k.on_clicked(lambda l: (st.update(K=int(l)), recompute(heavy=True)))
    radio_p.on_clicked(lambda l: (st.update(pmult=PMULTS[plabels.index(l)]), recompute(heavy=True)))
    radio_o.on_clicked(lambda l: (st.update(obj=l), recompute(heavy=True)))
    radio_d.on_clicked(lambda l: (st.update(disp=l), redraw()))
    check.on_clicked(on_check)
    for nm, fn in [("button_press_event", on_press), ("motion_notify_event", on_motion),
                   ("button_release_event", on_release),
                   ("key_press_event", lambda e: plt.close(fig) if e.key == "q" else None)]:
        fig.canvas.mpl_connect(nm, fn)
    plt.show()


if __name__ == "__main__":
    if SELFTEST:
        selftest()
    elif SNAPSHOT:
        snapshot()
    else:
        interactive()
