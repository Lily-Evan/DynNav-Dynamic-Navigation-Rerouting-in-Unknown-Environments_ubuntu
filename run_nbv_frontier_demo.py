from build_world_context import build_bottleneck_context
from nbv_frontier_sampling import find_frontier_cells, filter_reachable_frontiers
from nbv_irreversibility_scoring import score_candidates, topk
import os

def main():
    # =========================
    # Experimental metadata
    # =========================
    seed = 0
    episode = 0
    step = 0   # one-shot NBV decision (can be looped later)

    # =========================
    # Build environment context
    # =========================
    ctx = build_bottleneck_context()

    frontier = find_frontier_cells(
        ctx.free_mask,
        ctx.unc_grid,
        unc_thresh=0.6,
        known_thresh=0.5
    )

    frontier = filter_reachable_frontiers(
        frontier,
        ctx.R,
        R_min=0.05
    )

    print(f"Base start={ctx.start} tauR={ctx.tauR}")
    print(f"Frontier candidates (reachable): {len(frontier)}")

    # =========================
    # Score frontier candidates
    # =========================
    df = score_candidates(
        frontier,
        ctx.unc_grid,
        ctx.I_world,
        ctx.R,
        alpha=0.8,
        beta=0.5,
        r_local=2
    )

    # =========================
    # Add temporal metadata
    # =========================
    df["seed"] = seed
    df["episode"] = episode
    df["step"] = step

    # =========================
    # Save snapshot (optional)
    # =========================
    df.to_csv("nbv_frontier_scores.csv", index=False)

    # =========================
    # Append to unified log
    # =========================
    out = "nbv_frontier_scores_log.csv"
    header = not os.path.exists(out)
    df.to_csv(out, mode="a", header=header, index=False)
    print(f"[OK] appended {len(df)} rows -> {out} (header={header})")

    # =========================
    # Top-K analysis (for inspection)
    # =========================
    top = topk(df, "score_IG_I_R", 10)
    top.to_csv("nbv_frontier_top10.csv", index=False)

    print("\nTop-10 Frontier NBV summary:")
    print(
        f"mean IG      = {top['IG'].mean():.3f}\n"
        f"mean I_local = {top['I_local'].mean():.3f}\n"
        f"mean R_local = {top['R_local'].mean():.3f}"
    )

if __name__ == "__main__":
    main()
