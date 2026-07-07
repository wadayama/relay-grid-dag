"""E2 -- selection on the optimized MI (small grid so the oracle is feasible).

Greedy on f(S) is near the exhaustive oracle, one-swap polish recovers it, and both
beat the received-power baseline scored under the same MI oracle. The grid is
small (6 candidates) because each f(S) is now an inner optimization, so the full
oracle C(6,K) is affordable (it is not on the 25-candidate grid).
"""
import _common as C


def main():
    plt = C._mpl()
    coords = C.grid_coords(nx=3, ny=2)                       # 6 candidates
    s, names, coords = C.build_candidate_scene("near", coords=coords)
    K = 2
    kw = dict(iters=150)

    g = C.select_greedy(s, names, K, **kw)
    o_idx, o_cap = C.select_exhaustive(s, names, K, **kw)
    sw_idx, sw_cap = C.swap_search(s, names, g, K, **kw)
    rp = C.select_received_power(s, names, K)
    g_cap = C.optimized_mi(s, "tx", [names[i] for i in g], "rx", **kw)
    rp_cap = C.optimized_mi(s, "tx", [names[i] for i in rp], "rx", **kw)

    print(f"received-power {rp}      cap={rp_cap:.2f}")
    print(f"greedy         {g}      cap={g_cap:.2f}")
    print(f"greedy+swap    {sw_idx}      cap={sw_cap:.2f}")
    print(f"oracle         {o_idx}      cap={o_cap:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    labels = ["recv-power", "greedy", "greedy\n+swap", "oracle"]
    vals = [rp_cap, g_cap, sw_cap, o_cap]
    axes[0].bar(labels, vals, color=["0.6", "tab:blue", "tab:green", "tab:orange"])
    axes[0].set_ylabel("spectral efficiency f(S) [bits/s/Hz]")
    axes[0].set_title(f"Selection on MI (K={K}, L=6)")
    axes[0].grid(axis="y", alpha=0.3)
    C.plot_grid(axes[1], coords, title="grid + selections")
    C.mark_selected(axes[1], coords, o_idx, color="tab:orange", label="oracle")
    C.mark_selected(axes[1], coords, g, color="tab:blue", label="greedy", marker="s")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    p = f"{C.OUT}/e2_selection_value.pdf"; fig.savefig(p); print("saved", p)
    return (o_cap + 1e-6 >= sw_cap >= 0.95 * o_cap) and (g_cap >= rp_cap - 1e-6)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
