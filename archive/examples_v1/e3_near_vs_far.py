"""E3 -- near- vs far-field: why the near-field merge channel is worth optimizing.

The near-field spherical-wave channel is high-rank (range + angle), so the direct
link alone carries spatial degrees of freedom and the optimized MI is far
above the rank-one far-field (plane-wave) baseline. Figure: effective rank of the
direct channel and the optimized MI (a fixed 2-relay subset), near vs far.
"""
import _common as C


def main():
    plt = C._mpl()
    sub_idx = [6, 18]
    rank, cap, base = {}, {}, {}
    for model in ("near", "far"):
        s, names, _ = C.build_candidate_scene(model)
        rank[model] = C.nfd.viz.effrank(s.channel("tx", "rx"))     # direct-channel eff. rank
        subset = [names[i] for i in sub_idx]
        base[model] = C.subset_mi(s, subset)                       # scalar-AF
        cap[model] = C.optimized_mi(s, "tx", subset, "rx", iters=200)  # optimized
        print(f"{model}: direct eff-rank={rank[model]:.2f}  "
              f"scalar-AF={base[model]:.2f}  optimized={cap[model]:.2f} bits")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    axes[0].bar(["near", "far"], [rank["near"], rank["far"]],
                color=["tab:blue", "0.6"])
    axes[0].set_ylabel("effective rank of $H_{sd}$"); axes[0].set_title("direct-channel rank")
    axes[0].grid(axis="y", alpha=0.3)
    x = range(2)
    axes[1].bar([i - 0.2 for i in x], [base["near"], base["far"]], width=0.4,
                label="scalar-AF", color="0.6")
    axes[1].bar([i + 0.2 for i in x], [cap["near"], cap["far"]], width=0.4,
                label="optimized", color="tab:blue")
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(["near", "far"])
    axes[1].set_ylabel("bits/s/Hz"); axes[1].set_title("spectral efficiency (2 relays)")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = f"{C.OUT}/e3_near_vs_far.pdf"; fig.savefig(p); print("saved", p)
    return rank["near"] > rank["far"] and cap["near"] > cap["far"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
