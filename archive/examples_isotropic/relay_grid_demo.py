"""Interactive demo for NF-PRG (Near-field Programmable Relay Grid) *selection*.

Companion to nearfield-dag's `interactive_relay.py` (which moves relays by
gradient). Here the relays are FIXED candidate sites on a grid; dragging the Tx
re-runs the selection in real time and shows *which* K candidates the smart
(greedy-MI) rule activates versus the naive received-power rule -- the core
NF-PRG story: a few well-chosen relays beat naive magnitude-based selection, and
the right choice is non-obvious and geometry-dependent.

    Scene (Tx, Rx, L candidate relays)
      -> greedy-MI / received-power selection of K     [relay_grid_dag]
      -> mi(multirelay_merge) per candidate subset     [gaussian-dag K-recursion]
    Background: carrier-power field of the greedy-selected relays (relay_grid_dag.viz).

Controls:
    drag Tx (red triangle)   move the transmitter
    radio buttons (left)     number of activated relays K (1-4)
    q                        quit

Run (window):     uv run python examples/relay_grid_demo.py
Headless check:   uv run python examples/relay_grid_demo.py --selftest
Shareable still:  uv run python examples/relay_grid_demo.py --snapshot
"""
from __future__ import annotations

import os
import sys

import matplotlib
SELFTEST = "--selftest" in sys.argv
SNAPSHOT = "--snapshot" in sys.argv
if SELFTEST or SNAPSHOT:
    matplotlib.use("Agg")            # headless: identical on every platform
# else: let matplotlib auto-select the platform's interactive GUI backend
# (macosx / Qt / Tk), so the live window works for anyone who cloned the repo,
# not only on macOS. The window needs a desktop GUI backend; if none is available
# (e.g. a headless server), use --snapshot / --selftest instead.
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
GREEDY_COL = "#ff2d55"          # selected (greedy-MI) relays
F_C, S_SC, DF = 10.0, 8, 0.4    # OFDM band for the per-subcarrier rate inset
_K_WAVES = None                 # lazily filled (needs relay_grid_dag imported)

_gx, _gy = np.meshgrid(np.linspace(*PLANE_X, HEAT_NX), np.linspace(*PLANE_Y, HEAT_NY))
_GRID = torch.tensor(np.stack([_gx.ravel(), _gy.ravel()], axis=1), dtype=rgd.RDTYPE)
_COORDS = rgd.grid_coords(GRID_NX, GRID_NY, GRID_X, GRID_Y)


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


def selections(scene, names, K, *, wideband=True, P_relay=P_RELAY):
    """Return (greedy_idx, recvpow_idx, dict of sum-rates) for the current
    geometry. ``wideband`` picks the greedy criterion: the true OFDM sum-rate
    ``sum_k I_k`` (committed on drop) or a fast single-carrier proxy (live drag).
    Reported values are always the wideband sum-rate of the selected set.
    ``P_relay`` is the per-relay output power (relay-vs-Tx power knob)."""
    g_idx = (select_greedy_sumrate(scene, names, K, P_relay) if wideband
             else rgd.select_greedy_mi(scene, names, K, P_relay=P_relay))
    r_idx = rgd.select_received_power(scene, names, K)
    sr = dict(
        greedy=sum_rate(scene, names, g_idx, P_relay),
        recv=sum_rate(scene, names, r_idx, P_relay),
        direct=sum_rate(scene, names, [], P_relay),
    )
    return g_idx, r_idx, sr


def field_image(scene, names, g_idx, P_relay=P_RELAY):
    sel = [names[i] for i in g_idx]
    db = viz.carrier_field(scene, "tx", "rx", _GRID, relays=sel,
                           P_tx=P_RELAY, P_relay=P_relay)
    return db.reshape(HEAT_NY, HEAT_NX)


def per_sc_rates(scene, names, g_idx, P_relay=P_RELAY):
    """Per-subcarrier MI I_k of the (band-shared) selected relay set."""
    global _K_WAVES
    if _K_WAVES is None:
        _K_WAVES = rgd.subcarrier_wavenumbers(F_C, S_SC, DF)
    sel = [names[i] for i in g_idx]
    return np.array([rgd.subset_mi(scene, sel, k_wave=kw, P_relay=P_relay)
                     for kw in _K_WAVES])


def sum_rate(scene, names, idx, P_relay=P_RELAY):
    """Wideband simple sum-rate ``sum_k I_k`` (unit weights) of the selected set."""
    return float(per_sc_rates(scene, names, idx, P_relay).sum())


def select_greedy_sumrate(scene, names, K, P_relay=P_RELAY):
    """Forward greedy that maximizes the wideband sum-rate ``sum_k I_k`` (the S4
    OFDM criterion): the activation set ``a`` is shared across all subcarriers."""
    global _K_WAVES
    if _K_WAVES is None:
        _K_WAVES = rgd.subcarrier_wavenumbers(F_C, S_SC, DF)
    chosen, remaining = [], set(range(len(names)))
    for _ in range(K):
        best_j, best = None, -np.inf
        for j in remaining:
            sel = [names[i] for i in chosen + [j]]
            val = sum(rgd.subset_mi(scene, sel, k_wave=kw, P_relay=P_relay)
                      for kw in _K_WAVES)
            if val > best:
                best, best_j = val, j
        chosen.append(best_j)
        remaining.discard(best_j)
    return sorted(chosen)


def _add_sc_inset(ax, rates):
    """Small per-subcarrier rate bar chart inset; returns (axes, bar_container)."""
    axsc = ax.inset_axes([0.63, 0.005, 0.34, 0.20])
    axsc.set_facecolor((0, 0, 0, 0.35))           # dark translucent so white text reads
    bars = axsc.bar(range(1, S_SC + 1), rates, color=GREEDY_COL, width=0.82)
    axsc.set_title(r"per-subcarrier rate $I_k$", fontsize=8, color="white")
    axsc.set_xticks([]); axsc.tick_params(labelsize=6, colors="white")
    axsc.set_ylim(0, float(rates.max()) * 1.25 + 1e-6)
    for s in axsc.spines.values():
        s.set_edgecolor("white")
    return axsc, bars


def _update_sc_inset(axsc, bars, rates):
    for rect, h in zip(bars, rates):
        rect.set_height(float(h))
    axsc.set_ylim(0, float(rates.max()) * 1.25 + 1e-6)


def _info(sr, K, prelay_mult=1):
    return (f"K activated   = {K} of {len(_COORDS)}\n"
            f"relay power   = {prelay_mult}x Tx\n"
            f"sum-rate ΣI_k = {sr['greedy']:7.2f} bits/s/Hz ({S_SC} sc)\n"
            f"direct only   = {sr['direct']:7.2f} bits/s/Hz\n"
            f"relay gain    = {sr['greedy'] - sr['direct']:+7.2f} over direct")


def _draw_static(ax, *, draw_rx=True):
    """Candidate dots (and, by default, a static Rx marker)."""
    ax.scatter(_COORDS[:, 0], _COORDS[:, 1], s=70, c="0.75", edgecolors="k",
               linewidths=0.4, zorder=3, label="candidate relays")
    if draw_rx:
        ax.plot([RX_C[0]], [RX_C[1]], "cv", ms=15, zorder=5, label="Rx")


# --------------------------------- self-test ---------------------------------
def selftest():
    print("selftest: NF-PRG selection demo over a few Tx positions ...")
    txs = [(0.0, 0.0), (0.0, 6.0), (-3.0, -4.0)]
    fig, axes = plt.subplots(1, len(txs), figsize=(5.0 * len(txs), 4.4))
    K = 3
    wins = 0
    for ax, tx in zip(axes, txs):
        s, names = build_scene(tx)
        g_idx, r_idx, mi = selections(s, names, K)
        assert np.isfinite(mi["greedy"]) and mi["greedy"] > 0
        assert mi["greedy"] >= mi["recv"] - 1e-6, "greedy below received-power"
        wins += mi["greedy"] > mi["recv"] + 1e-6
        ax.imshow(field_image(s, names, g_idx), origin="lower",
                  extent=[*PLANE_X, *PLANE_Y], aspect="auto", cmap="inferno",
                  vmin=-40, vmax=0, zorder=0)
        _draw_static(ax)
        ax.scatter(_COORDS[g_idx, 0], _COORDS[g_idx, 1], s=240, facecolors="none",
                   edgecolors=GREEDY_COL, linewidths=2.6, zorder=7)
        ax.plot([tx[0]], [tx[1]], "r^", ms=14, zorder=8)
        ax.set_xlim(*PLANE_X); ax.set_ylim(*PLANE_Y)
        ax.set_title(f"Tx={tx}: greedy MI {mi['greedy']:.1f} (direct {mi['direct']:.1f})")
        print(f"  Tx={tx}: greedy={mi['greedy']:.2f} recv={mi['recv']:.2f} "
              f"direct={mi['direct']:.2f}  greedy_set={g_idx} recv_set={r_idx}")
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "relay_grid_demo_selftest.png"), dpi=110)
    print(f"  greedy strictly beat received-power in {wins}/{len(txs)} positions")
    print(f"  saved {out}/relay_grid_demo_selftest.png\nselftest passed.")


# --------------------------------- snapshot ----------------------------------
def snapshot():
    """A shareable still: off-axis Tx where smart vs naive selection visibly differ."""
    from matplotlib.widgets import RadioButtons
    K = 3
    tx = np.array([-2.0, 5.0])
    s, names = build_scene(tx)
    g_idx, r_idx, mi = selections(s, names, K)

    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    plt.subplots_adjust(left=0.16)
    ax.set_xlim(*PLANE_X); ax.set_ylim(*PLANE_Y)
    ax.set_xlabel(r"x [$\lambda$]"); ax.set_ylabel(r"y [$\lambda$]")
    im = ax.imshow(field_image(s, names, g_idx), origin="lower",
                   extent=[*PLANE_X, *PLANE_Y], aspect="auto", cmap="inferno",
                   vmin=-40, vmax=0, zorder=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("carrier power [dB] (Tx + greedy-selected relays)")
    _draw_static(ax)
    for i in g_idx:
        ax.plot([tx[0], _COORDS[i, 0], RX_C[0]], [tx[1], _COORDS[i, 1], RX_C[1]],
                color=GREEDY_COL, lw=1.3, alpha=0.8, zorder=4)
    ax.scatter(_COORDS[g_idx, 0], _COORDS[g_idx, 1], s=300, facecolors="none",
               edgecolors=GREEDY_COL, linewidths=2.8, zorder=7,
               label="greedy-MI selected relays")
    ax.plot([tx[0]], [tx[1]], "r^", ms=16, zorder=8, label="Tx (drag me)")
    ax.text(0.02, 0.98, _info(mi, K), transform=ax.transAxes, va="top",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    _add_sc_inset(ax, per_sc_rates(s, names, g_idx))
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("NF-PRG: drag the Tx — a few relays are *selected* from the grid "
                 "to shape the near-field channel")
    ax_radio = fig.add_axes([0.015, 0.46, 0.10, 0.32])
    ax_radio.set_title("K (0 = direct)", fontsize=9)
    RadioButtons(ax_radio, [str(k) for k in range(0, MAX_K + 1)], active=K)

    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "relay_grid_demo.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}  (K={K}, greedy={mi['greedy']:.2f}, recv={mi['recv']:.2f})")


# -------------------------------- interactive --------------------------------
def interactive():
    from matplotlib.widgets import RadioButtons

    st = {"tx": np.array([0.0, 0.0]), "rx": np.array(RX_C, dtype=float),
          "drag": None, "K": 3, "pmult": 1}
    P = lambda: st["pmult"] * P_RELAY     # current per-relay output power
    s, names = build_scene(st["tx"])

    matplotlib.rcParams["toolbar"] = "None"      # no pan/zoom rubber-band on drag
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    plt.subplots_adjust(left=0.16)
    ax.set_xlim(*PLANE_X); ax.set_ylim(*PLANE_Y)
    ax.set_xlabel(r"x [$\lambda$]"); ax.set_ylabel(r"y [$\lambda$]")

    g_idx, r_idx, mi = selections(s, names, st["K"], P_relay=P())
    im = ax.imshow(field_image(s, names, g_idx, P()), origin="lower",
                   extent=[*PLANE_X, *PLANE_Y], aspect="auto", cmap="inferno",
                   vmin=-40, vmax=0, zorder=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("carrier power [dB] (Tx + greedy-selected relays)")
    _draw_static(ax, draw_rx=False)
    greedy_sc = ax.scatter(_COORDS[g_idx, 0], _COORDS[g_idx, 1], s=300,
                           facecolors="none", edgecolors=GREEDY_COL, linewidths=2.8,
                           zorder=7, label="greedy-MI selected relays")
    path_lns = [ax.plot([], [], color=GREEDY_COL, lw=1.2, alpha=0.8, zorder=4)[0]
                for _ in range(MAX_K)]
    tx_mk, = ax.plot([st["tx"][0]], [st["tx"][1]], "r^", ms=16, zorder=8,
                     label="Tx (drag me)")
    rx_mk, = ax.plot([st["rx"][0]], [st["rx"][1]], "cv", ms=16, zorder=8,
                     label="Rx (drag me)")
    txt = ax.text(0.02, 0.98, _info(mi, st["K"], st["pmult"]), transform=ax.transAxes,
                  va="top", fontsize=10, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    axsc, sc_bars = _add_sc_inset(ax, per_sc_rates(s, names, g_idx, P()))
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("NF-PRG: drag Tx or Rx — relays are selected from the grid")

    ax_radio = fig.add_axes([0.015, 0.52, 0.10, 0.30])
    ax_radio.set_title("K (0 = direct)", fontsize=9)
    radio = RadioButtons(ax_radio, [str(k) for k in range(0, MAX_K + 1)], active=st["K"])

    PMULTS = [1, 10, 100]
    plabels = [f"{m}x" for m in PMULTS]
    ax_pwr = fig.add_axes([0.015, 0.16, 0.10, 0.24])
    ax_pwr.set_title("relay power\n(x Tx)", fontsize=9)
    radio_p = RadioButtons(ax_pwr, plabels, active=PMULTS.index(st["pmult"]))

    def recompute(heavy):
        """Light path (markers/selection/text) always; heavy path (per-subcarrier
        inset + carrier field) only on drop -- keeps dragging cheap and avoids the
        full-canvas churn that produced redraw artifacts."""
        s.move("tx", list(st["tx"])); s.move("rx", list(st["rx"]))
        g_idx, r_idx, mi = selections(s, names, st["K"], wideband=heavy, P_relay=P())
        greedy_sc.set_offsets(_COORDS[g_idx].reshape(-1, 2))
        for k, ln in enumerate(path_lns):
            if k < len(g_idx):
                i = g_idx[k]
                ln.set_data([st["tx"][0], _COORDS[i, 0], st["rx"][0]],
                            [st["tx"][1], _COORDS[i, 1], st["rx"][1]])
            else:
                ln.set_data([], [])
        tx_mk.set_data([st["tx"][0]], [st["tx"][1]])
        rx_mk.set_data([st["rx"][0]], [st["rx"][1]])
        txt.set_text(_info(mi, st["K"], st["pmult"]))
        if heavy:
            _update_sc_inset(axsc, sc_bars, per_sc_rates(s, names, g_idx, P()))
            im.set_data(field_image(s, names, g_idx, P()))
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
            recompute(heavy=False)               # cheap during drag

    def on_release(e):
        if st["drag"]:
            st["drag"] = None
            recompute(heavy=True)                # inset + heatmap on drop

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
