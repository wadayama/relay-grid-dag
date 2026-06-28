"""Interactive demo for the precoded Near-field Relay Grid.

Drag the Tx (red triangle) or Rx (cyan triangle): a few candidate relays are
activated and the Tx / relay precoders are optimized for the shown configuration.
The carrier-field background is rendered from those same optimized precoders, so
the field and the reported MI describe one transmit configuration.

UI mirrors the earlier selection demo (K radio, relay-power radio, drag Tx/Rx,
Tx->relay->Rx path lines, info box, light-while-dragging / heavy-on-drop). On drop
the relays are selected by greedy on the optimized MI f(S) and the Tx/relay
precoders are optimized; the carrier field is rendered from those precoders. During
a drag only the terminals move (the configuration re-solves on release).
(The OFDM per-subcarrier inset of the earlier demo is omitted.)

Controls:
    drag Tx (red triangle) / Rx (cyan triangle)   move the terminals
    radio (left)   number of activated relays K (0 = direct)
    radio (left)   relay power (dB rel. Tx)
    q              quit

    uv run --extra examples python examples/relay_grid_demo.py            # window
    uv run --extra examples python examples/relay_grid_demo.py --snapshot # -> out/
    uv run --extra examples python examples/relay_grid_demo.py --selftest # headless
"""
from __future__ import annotations

import os
import sys

import matplotlib
SELFTEST = "--selftest" in sys.argv
SNAPSHOT = "--snapshot" in sys.argv
if SELFTEST or SNAPSHOT:
    matplotlib.use("Agg")            # headless: identical on every platform
import matplotlib.pyplot as plt
import numpy as np
import torch

import relay_grid_dag as rgd
from relay_grid_dag import viz

# ------------------------------- configuration -------------------------------
RX_C = (20.0, 0.0)
PLANE_X = (-5.0, 25.0)
PLANE_Y = (-10.0, 10.0)
GRID_X = (5.0, 15.0)
GRID_Y = (-6.0, 6.0)
GRID_NX = GRID_NY = 5            # L = 25 candidate sites
DIRECT_ATTEN = 0.4              # partially blocked direct -> relays carry value
N_TX = N_RX = 8                 # Tx / Rx antennas (uniform linear arrays)
N_RELAY = 4                     # antennas per candidate relay
P_RELAY = 100.0                 # per-relay output power (Tx power = 100)
MAX_K = 4
HEAT_NX, HEAT_NY = 150, 110
CMAP = "viridis"                # background carrier-field colormap
SEL_COL = "#ff2bd6"             # activated relays / relay path lines (magenta: pops on viridis)
DIRECT_COL = "#ffffff"          # weak direct Tx-Rx link (white dashed)
SEL_ITERS = 5                  # greedy-selection inner-optimization depth (on drop)
DISP_ITERS = 50                # precoder-optimization depth for the shown set

_gx, _gy = np.meshgrid(np.linspace(*PLANE_X, HEAT_NX), np.linspace(*PLANE_Y, HEAT_NY))
_GRID = torch.tensor(np.stack([_gx.ravel(), _gy.ravel()], axis=1), dtype=rgd.RDTYPE)
_COORDS = rgd.grid_coords(GRID_NX, GRID_NY, GRID_X, GRID_Y)

SPECS = (f"near-field LoS, single pair  |  Tx/Rx: {N_TX}-elem ULA, relay: {N_RELAY}-elem ULA "
         f"(0.5λ spacing)  |  {GRID_NX}x{GRID_NY} relay grid, "
         f"{(GRID_X[1]-GRID_X[0])/(GRID_NX-1):.1f}x{(GRID_Y[1]-GRID_Y[0])/(GRID_NY-1):.1f}λ "
         f"(L={GRID_NX*GRID_NY})  |  P_tx=P_relay  |  direct atten {DIRECT_ATTEN}")


def build_scene(tx_c):
    """Scene with a (drag-movable) Tx, fixed Rx, and the fixed candidate grid."""
    s = rgd.Scene(model="near", direct_atten=DIRECT_ATTEN, spacing=0.5)
    s.add_node("tx", list(tx_c), N_TX, "source")
    s.add_node("rx", list(RX_C), N_RX, "sink")
    names = []
    for i, (x, y) in enumerate(_COORDS):
        s.add_node(f"c{i}", [float(x), float(y)], N_RELAY, "relay")
        names.append(f"c{i}")
    return s, names


def select_set(scene, names, K, P_relay):
    """Greedy activation of K relays on the optimized MI f(S)."""
    if K <= 0:
        return []
    return rgd.select_greedy(scene, names, K, iters=SEL_ITERS, P_relay=P_relay)


def optimize_set(scene, names, idx, P_relay):
    """Optimize the Tx/relay precoders for the chosen set -> (MI, F, W)."""
    sel = [names[i] for i in idx]
    return rgd.optimize_precoders(scene, "tx", sel, "rx",
                                  P_relay=P_relay, iters=DISP_ITERS)


def field_image(scene, names, idx, F, W):
    sel = [names[i] for i in idx]
    db = viz.carrier_field(scene, "tx", _GRID, F=F, relays=sel, relay_mats=W)
    return db.reshape(HEAT_NY, HEAT_NX)


def _info(K, direct, *, cap=None, pmult=1):
    n = len(_COORDS)
    if cap is None:
        capline = "spectral eff  =  (release to optimize)"
        gainline = "relay gain    =  (release to optimize)"
    else:
        capline = f"spectral eff  = {cap:7.2f} bits/s/Hz (optimized)"
        gainline = f"relay gain    = {cap - direct:+7.2f} over direct"
    return (f"K activated   = {K} of {n}\n"
            f"relay power   = +{int(round(10 * np.log10(pmult)))} dB rel. Tx\n"
            f"{capline}\n"
            f"direct only   = {direct:7.2f} bits/s/Hz\n"
            f"{gainline}")


def _draw_static(ax, *, draw_rx=True):
    ax.scatter(_COORDS[:, 0], _COORDS[:, 1], s=70, c="0.75", edgecolors="k",
               linewidths=0.4, zorder=3, label="candidate relays")
    if draw_rx:
        ax.plot([RX_C[0]], [RX_C[1]], "cv", ms=15, zorder=5, label="Rx")


def _render_field(scene, names, idx, P_relay):
    """Heavy step: optimize precoders for ``idx`` and return (cap, field_image)."""
    if idx:
        cap, F, W = optimize_set(scene, names, idx, P_relay)
        return cap, field_image(scene, names, idx, F, W)
    cap, F, W = rgd.optimize_precoders(scene, "tx", [], "rx",
                                       P_relay=P_relay, iters=DISP_ITERS)
    return cap, field_image(scene, names, [], F, W)


# --------------------------------- self-test ---------------------------------
def selftest():
    print("selftest: precoded relay-grid demo over a few Tx positions ...")
    txs = [(0.0, 0.0), (0.0, 6.0), (-3.0, -4.0)]
    fig, axes = plt.subplots(1, len(txs), figsize=(5.0 * len(txs), 4.4))
    K = 3
    for ax, tx in zip(axes, txs):
        s, names = build_scene(tx)
        idx = select_set(s, names, K, P_RELAY)
        direct = rgd.direct_only_mi(s, P_relay=P_RELAY)
        cap, fld = _render_field(s, names, idx, P_RELAY)
        assert np.isfinite(cap) and cap >= direct - 1e-6
        ax.imshow(fld, origin="lower", extent=[*PLANE_X, *PLANE_Y], aspect="auto",
                  cmap=CMAP, vmin=-40, vmax=0, zorder=0)
        _draw_static(ax)
        ax.plot([tx[0], RX_C[0]], [tx[1], RX_C[1]], color=DIRECT_COL, lw=1.1,
                ls="--", alpha=0.9, zorder=6)
        for i in idx:
            ax.plot([tx[0], _COORDS[i, 0], RX_C[0]], [tx[1], _COORDS[i, 1], RX_C[1]],
                    color=SEL_COL, lw=1.4, alpha=0.9, zorder=6)
        ax.scatter(_COORDS[idx, 0], _COORDS[idx, 1], s=240, facecolors="none",
                   edgecolors=SEL_COL, linewidths=2.6, zorder=7)
        ax.plot([tx[0]], [tx[1]], "r^", ms=14, zorder=8)
        ax.set_xlim(*PLANE_X); ax.set_ylim(*PLANE_Y)
        ax.set_title(f"Tx={tx}: cap {cap:.1f} (direct {direct:.1f})")
        print(f"  Tx={tx}: cap={cap:.2f} direct={direct:.2f} set={idx}")
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "relay_grid_demo_selftest.png"), dpi=110)
    print(f"  saved {out}/relay_grid_demo_selftest.png\nselftest passed.")


# --------------------------------- snapshot ----------------------------------
def snapshot():
    from matplotlib.widgets import RadioButtons
    K = 3
    tx = np.array([-2.0, 5.0])
    s, names = build_scene(tx)
    idx = select_set(s, names, K, P_RELAY)
    direct = rgd.direct_only_mi(s, P_relay=P_RELAY)
    cap, fld = _render_field(s, names, idx, P_RELAY)

    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    plt.subplots_adjust(left=0.16)
    ax.set_xlim(*PLANE_X); ax.set_ylim(*PLANE_Y)
    ax.set_xlabel(r"x [$\lambda$]"); ax.set_ylabel(r"y [$\lambda$]")
    im = ax.imshow(fld, origin="lower", extent=[*PLANE_X, *PLANE_Y], aspect="auto",
                   cmap=CMAP, vmin=-40, vmax=0, zorder=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("carrier field [dB] (optimized Tx + relay precoders)")
    _draw_static(ax)
    ax.plot([tx[0], RX_C[0]], [tx[1], RX_C[1]], color=DIRECT_COL, lw=1.4,
            ls="--", alpha=0.9, zorder=4, label="direct link (weak)")
    for i in idx:
        ax.plot([tx[0], _COORDS[i, 0], RX_C[0]], [tx[1], _COORDS[i, 1], RX_C[1]],
                color=SEL_COL, lw=1.6, alpha=0.9, zorder=4)
    ax.scatter(_COORDS[idx, 0], _COORDS[idx, 1], s=300, facecolors="none",
               edgecolors=SEL_COL, linewidths=2.8, zorder=7, label="activated relays")
    ax.plot([tx[0]], [tx[1]], "r^", ms=16, zorder=8, label="Tx (drag me)")
    ax.text(0.02, 0.98, _info(K, direct, cap=cap), transform=ax.transAxes,
            va="top", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("NRG: drag the Tx — relays are activated and precoders optimized "
                 "to shape the near-field channel")
    fig.text(0.5, 0.012, SPECS, ha="center", va="bottom", fontsize=8, color="0.35")
    ax_radio = fig.add_axes([0.015, 0.46, 0.10, 0.32])
    ax_radio.set_title("K (0 = direct)", fontsize=9)
    RadioButtons(ax_radio, [str(k) for k in range(0, MAX_K + 1)], active=K)
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "relay_grid_demo.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}  (K={K}, SE={cap:.2f})")


# -------------------------------- interactive --------------------------------
def interactive():
    from matplotlib.widgets import RadioButtons

    st = {"tx": np.array([0.0, 0.0]), "rx": np.array(RX_C, dtype=float),
          "drag": None, "K": 3, "pmult": 1}
    P = lambda: st["pmult"] * P_RELAY
    s, names = build_scene(st["tx"])

    matplotlib.rcParams["toolbar"] = "None"      # no pan/zoom rubber-band on drag
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    plt.subplots_adjust(left=0.16)
    ax.set_xlim(*PLANE_X); ax.set_ylim(*PLANE_Y)
    ax.set_xlabel(r"x [$\lambda$]"); ax.set_ylabel(r"y [$\lambda$]")

    idx = select_set(s, names, st["K"], P())
    st["idx"] = idx
    direct = rgd.direct_only_mi(s, P_relay=P())
    cap, fld = _render_field(s, names, idx, P())
    im = ax.imshow(fld, origin="lower", extent=[*PLANE_X, *PLANE_Y], aspect="auto",
                   cmap=CMAP, vmin=-40, vmax=0, zorder=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("carrier field [dB] (optimized Tx + relay precoders)")
    _draw_static(ax, draw_rx=False)
    sel_sc = ax.scatter(_COORDS[idx, 0], _COORDS[idx, 1], s=300, facecolors="none",
                        edgecolors=SEL_COL, linewidths=2.8, zorder=7,
                        label="activated relays")
    path_lns = [ax.plot([], [], color=SEL_COL, lw=1.6, alpha=0.9, zorder=4)[0]
                for _ in range(MAX_K)]
    direct_ln, = ax.plot([st["tx"][0], st["rx"][0]], [st["tx"][1], st["rx"][1]],
                         color=DIRECT_COL, lw=1.4, ls="--", alpha=0.9, zorder=4,
                         label="direct link (weak)")
    tx_mk, = ax.plot([st["tx"][0]], [st["tx"][1]], "r^", ms=16, zorder=8,
                     label="Tx (drag me)")
    rx_mk, = ax.plot([st["rx"][0]], [st["rx"][1]], "cv", ms=16, zorder=8,
                     label="Rx (drag me)")
    txt = ax.text(0.02, 0.98, _info(st["K"], direct, cap=cap, pmult=st["pmult"]),
                  transform=ax.transAxes, va="top", fontsize=10, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("NRG: drag Tx or Rx — relays selected on MI, precoders optimized")
    fig.text(0.5, 0.012, SPECS, ha="center", va="bottom", fontsize=8, color="0.35")

    ax_radio = fig.add_axes([0.015, 0.52, 0.10, 0.30])
    ax_radio.set_title("K (0 = direct)", fontsize=9)
    radio = RadioButtons(ax_radio, [str(k) for k in range(0, MAX_K + 1)], active=st["K"])

    PMULTS = [1, 10, 100]
    plabels = [f"+{int(round(10 * np.log10(m)))} dB" for m in PMULTS]
    ax_pwr = fig.add_axes([0.015, 0.16, 0.10, 0.24])
    ax_pwr.set_title("relay power\n(dB rel. Tx)", fontsize=9)
    radio_p = RadioButtons(ax_pwr, plabels, active=PMULTS.index(st["pmult"]))

    def recompute(heavy):
        """Light path (move terminals; keep the current selection) while dragging;
        heavy path (re-select on MI + optimize precoders + carrier field) only
        on drop / radio change -- so selection and field are always on the optimized
        MI, and dragging stays cheap."""
        s.move("tx", list(st["tx"])); s.move("rx", list(st["rx"]))
        if heavy:
            st["idx"] = select_set(s, names, st["K"], P())
        idx = st["idx"]
        direct = rgd.direct_only_mi(s, P_relay=P())
        sel_sc.set_offsets(_COORDS[idx].reshape(-1, 2) if idx else np.empty((0, 2)))
        for k, ln in enumerate(path_lns):
            if k < len(idx):
                i = idx[k]
                ln.set_data([st["tx"][0], _COORDS[i, 0], st["rx"][0]],
                            [st["tx"][1], _COORDS[i, 1], st["rx"][1]])
            else:
                ln.set_data([], [])
        tx_mk.set_data([st["tx"][0]], [st["tx"][1]])
        rx_mk.set_data([st["rx"][0]], [st["rx"][1]])
        direct_ln.set_data([st["tx"][0], st["rx"][0]], [st["tx"][1], st["rx"][1]])
        if heavy:
            cap, fld = _render_field(s, names, idx, P())
            im.set_data(fld)
            txt.set_text(_info(st["K"], direct, cap=cap, pmult=st["pmult"]))
        else:
            txt.set_text(_info(st["K"], direct, cap=None, pmult=st["pmult"]))
        fig.canvas.draw_idle()

    def _near(e, key):
        return abs(e.xdata - st[key][0]) < 2 and abs(e.ydata - st[key][1]) < 2

    def on_press(e):
        if e.inaxes is ax and e.xdata is not None:
            if _near(e, "tx"):
                st["drag"] = "tx"
            elif _near(e, "rx"):
                st["drag"] = "rx"

    def on_motion(e):
        if st["drag"] and e.inaxes is ax and e.xdata is not None:
            st[st["drag"]] = np.array([np.clip(e.xdata, *PLANE_X),
                                       np.clip(e.ydata, *PLANE_Y)])
            recompute(heavy=False)

    def on_release(e):
        if st["drag"]:
            st["drag"] = None
            recompute(heavy=True)

    def on_key(e):
        if e.key == "q":
            plt.close(fig)

    radio.on_clicked(lambda label: (st.update(K=int(label)), recompute(heavy=True)))
    radio_p.on_clicked(lambda label: (st.update(pmult=PMULTS[plabels.index(label)]),
                                      recompute(heavy=True)))
    for nm, fn in [("button_press_event", on_press),
                   ("motion_notify_event", on_motion),
                   ("button_release_event", on_release),
                   ("key_press_event", on_key)]:
        fig.canvas.mpl_connect(nm, fn)
    plt.show()


if __name__ == "__main__":
    if SELFTEST:
        selftest()
    elif SNAPSHOT:
        snapshot()
    else:
        interactive()
