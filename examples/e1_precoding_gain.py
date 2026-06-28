"""E1 -- precoding gain: optimized MI f(S) vs the scalar-AF baseline.

The differentiable joint Tx/relay precoding engine (SPEC.md Sec. 4): for a fixed
activated subset, optimizing the Tx precoder F and the relay matrices {W_l} raises
the mutual information well above the scalar-AF / isotropic baseline (which is
the conventional relay, no optimization). Figure: MI vs scalar-AF across active-set size.
"""
import _common as C


def main():
    plt = C._mpl()
    s, names, coords = C.build_candidate_scene("near")
    pool = [6, 18, 11, 1]                       # a fixed nested sequence of relays
    Ks, base, opt = [], [], []
    for k in range(len(pool) + 1):
        subset = [names[i] for i in pool[:k]]
        Ks.append(k)
        base.append(C.subset_mi(s, subset))                          # scalar-AF baseline
        opt.append(C.optimized_mi(s, "tx", subset, "rx", iters=200))     # optimized MI
    for k, b, o in zip(Ks, base, opt):
        print(f"K={k}: scalar-AF={b:6.2f}  optimized={o:6.2f}  gain={o - b:+5.2f} bits")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(Ks, base, "o--", color="0.5", label="scalar-AF baseline")
    ax.plot(Ks, opt, "s-", color="tab:blue", label="optimized MI")
    ax.set_xlabel("active relays K"); ax.set_ylabel("bits/s/Hz")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(Ks)
    fig.tight_layout()
    p = f"{C.OUT}/e1_precoding_gain.pdf"; fig.savefig(p); print("saved", p)
    return all(o >= b - 1e-6 for o, b in zip(opt, base))


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
