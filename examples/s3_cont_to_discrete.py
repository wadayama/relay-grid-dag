"""S3 — Does the continuous-to-discrete pipeline recover the oracle? (the method)

NF-PRG's proposed optimiser (note_v2 sec.4, cardboard guide offline section):
  1. optimise K *virtual* relay positions as continuous variables (position-grad PGA),
  2. round each to its nearest candidate grid point,
  3. polish with 1-swap local search,
and claims this approaches the exhaustive oracle at a fraction of the cost.

We run it for K=2 and report R_cont, R_round, R_swap, R_oracle (+ random mean).

PASS if rounding loss is small (R_round within ~12% of oracle), the swap never
hurts and lands within ~3% of oracle, and the whole chain beats random-mean.

Output: out/s3_cont_to_discrete.png
"""
import numpy as np
import torch

import _common as C

K = 2
STEP = 0.04
ITERS = 300
D_MIN = 1.8        # min separation between virtual relays (wavelengths)


def pga_continuous(seed=0):
    """Projected-gradient ascent on K virtual relay centres. Returns the final
    centres (K,2), the MI, and the trajectory list."""
    rng = np.random.default_rng(seed)
    s = C.build_candidate_scene("near")[0]
    lo = np.array([C.GRID_X[0], C.GRID_Y[0]])
    hi = np.array([C.GRID_X[1], C.GRID_Y[1]])
    vnames = []
    for i in range(K):
        start = lo + rng.random(2) * (hi - lo)
        s.add_node(f"v{i}", start.tolist(), C.N_RELAY, "relay", movable_node=True)
        vnames.append(f"v{i}")

    def centers_now():
        return np.array([s._nodes[v]["center"].detach().tolist() for v in vnames])

    traj = [centers_now()]
    for _ in range(ITERS):
        for c in s.params:
            if c.grad is not None:
                c.grad.zero_()
        I = C.nfd.mi(nfd_merge(s, vnames))
        I.backward()
        with torch.no_grad():
            stepped = [c.detach() + STEP * c.grad for c in s.params]
            boxed = [C.nfd.project_box(c, lo.tolist(), hi.tolist()) for c in stepped]
            sep = C.nfd.project_min_separation(boxed, D_MIN)   # list in, list out
        for v, c in zip(vnames, sep):
            s.move(v, c.tolist())
        traj.append(centers_now())
    with torch.no_grad():
        R_cont = C.nfd.mi(nfd_merge(s, vnames)).item()
    return centers_now(), R_cont, traj


def nfd_merge(scene, vnames):
    return C.nfd.multirelay_merge(scene, "tx", vnames, "rx", P_relay=C.P_RELAY)


def round_to_grid(final, coords):
    """Nearest candidate per virtual relay, resolving collisions to the next
    nearest free grid point."""
    idx = []
    for p in final:
        order = np.argsort(((coords - p) ** 2).sum(1))
        for cand in order:
            if cand not in idx:
                idx.append(int(cand))
                break
    return sorted(idx)


def main():
    plt = C._mpl()
    s, names, coords = C.build_candidate_scene("near")

    # best multi-start continuous solution
    best = None
    for seed in range(4):
        final, R_cont, traj = pga_continuous(seed)
        if best is None or R_cont > best[1]:
            best = (final, R_cont, traj)
    final, R_cont, traj = best

    round_idx = round_to_grid(final, coords)
    R_round = C.subset_mi(s, [names[i] for i in round_idx])
    swap_idx, R_swap = C.swap_search(s, names, round_idx, K)
    ora_idx, R_oracle = C.select_exhaustive(s, names, K)
    rnd = C.random_subset_stats(s, names, K, trials=300, seed=2)

    round_loss = (R_oracle - R_round) / R_oracle
    swap_gap = (R_oracle - R_swap) / R_oracle
    ok = (round_loss <= 0.12 and R_swap >= R_round - 1e-6
          and swap_gap <= 0.03 and R_swap > rnd["mean"])

    # ---- figure: pipeline on the candidate grid + a results bar
    fig = plt.figure(figsize=(12, 5.0))
    axm = fig.add_subplot(1, 2, 1)
    axb = fig.add_subplot(1, 2, 2)

    C.plot_grid(axm, coords, title="continuous PGA -> rounded -> swap -> oracle")
    traj = np.asarray(traj)                       # (T, K, 2)
    for r in range(K):
        axm.plot(traj[:, r, 0], traj[:, r, 1], "-", color="magenta", lw=1.0,
                 alpha=0.7, zorder=4)
    axm.scatter(final[:, 0], final[:, 1], marker="*", s=240, c="magenta",
                edgecolors="k", zorder=7, label="continuous PGA (virtual)")
    C.mark_selected(axm, coords, round_idx, color="tab:orange",
                    label="rounded", marker="s")
    C.mark_selected(axm, coords, swap_idx, color="blue", label="after 1-swap",
                    marker="D")
    C.mark_selected(axm, coords, ora_idx, color="red", label="oracle", marker="o")
    axm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
               fontsize=8, frameon=False)

    labels = ["random\n(mean)", "R_cont", "R_round", "R_swap", "R_oracle"]
    vals = [rnd["mean"], R_cont, R_round, R_swap, R_oracle]
    cols = ["0.7", "magenta", "tab:orange", "blue", "red"]
    bars = axb.bar(labels, vals, color=cols)
    axb.bar_label(bars, fmt="%.2f", fontsize=8)
    axb.errorbar(0, rnd["mean"], yerr=rnd["std"], fmt="none", ecolor="k", capsize=3)
    axb.set_ylabel("I(X;Y) [bits/s/Hz]")
    axb.set_title("continuous-to-discrete recovers the oracle")
    axb.grid(True, axis="y", lw=0.3, alpha=0.5)

    fig.suptitle("S3 — position-gradient PGA + grid rounding + 1-swap ~ exhaustive oracle",
                 fontsize=11)
    path = f"{C.OUT}/s3_cont_to_discrete.png"
    fig.savefig(path, bbox_inches="tight")

    print("=== S3: continuous-to-discrete ===")
    print(f"continuous PGA final (virtual): {np.round(final, 2).tolist()}")
    print(f"rounded indices  {round_idx}")
    print(f"swap indices     {swap_idx}")
    print(f"oracle indices   {ora_idx}")
    print(f"R_cont  = {R_cont:.3f}")
    print(f"R_round = {R_round:.3f}  (rounding loss vs oracle = {round_loss*100:.1f}%)")
    print(f"R_swap  = {R_swap:.3f}  (gap vs oracle = {swap_gap*100:.1f}%)")
    print(f"R_oracle= {R_oracle:.3f}")
    print(f"random mean = {rnd['mean']:.3f} +- {rnd['std']:.3f}")
    print(f"saved {path}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
