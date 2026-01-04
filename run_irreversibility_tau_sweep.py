# run_irreversibility_tau_sweep.py
import numpy as np
import pandas as pd

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
)
from irreversibility_planner import astar_irreversibility_constrained


def pick_start_goal_near_I(
    free_mask: np.ndarray,
    I_grid: np.ndarray,
    I_target: float = 0.80,
    band: float = 0.05,
):
    """
    Pick fixed start/goal from cells with I near a target value (within ±band).
    This yields a meaningful tau sweep (not trivially blocked by start/goal).

    Returns: (start, goal, used_band)
    """
    candidates = free_mask & (I_grid >= (I_target - band)) & (I_grid <= (I_target + band))
    pts = np.argwhere(candidates)

    # If not enough, widen band gradually
    used_band = band
    if pts.shape[0] < 2:
        for b in [0.08, 0.10, 0.15, 0.20]:
            candidates = free_mask & (I_grid >= (I_target - b)) & (I_grid <= (I_target + b))
            pts = np.argwhere(candidates)
            if pts.shape[0] >= 2:
                used_band = b
                break

    if pts.shape[0] < 2:
        raise RuntimeError(
            f"Could not find enough cells near I_target={I_target:.2f}. "
            f"Try a different I_target (e.g. 0.6 or 0.7)."
        )

    pts_list = [tuple(map(int, p)) for p in pts]

    # Choose two far-apart points (approx by subsampling if too many)
    if len(pts_list) > 2000:
        step = max(1, len(pts_list) // 2000)
        pts_list = pts_list[::step]

    start = pts_list[0]
    goal = pts_list[-1]
    best_d = -1

    # coarse search for farthest pair
    step2 = max(1, len(pts_list) // 200)
    for a in pts_list[::step2]:
        for bpt in pts_list[::step2]:
            d = abs(a[0] - bpt[0]) + abs(a[1] - bpt[1])
            if d > best_d:
                best_d = d
                start, goal = a, bpt

    return start, goal, used_band


def main():
    path_csv = "coverage_grid_with_uncertainty.csv"

    # Load uncertainty grid[row,col]
    unc_grid = load_grid_from_cell_table_csv(
        path_csv,
        value_col="uncertainty",
        row_col=("row", "col"),
        fill_nan_with="max",
    )
    free_mask = np.isfinite(unc_grid)

    # Feature density proxy (demo): higher where uncertainty is low
    unc01 = (unc_grid - unc_grid.min()) / (unc_grid.max() - unc_grid.min() + 1e-9)
    feat_density = 1.0 - unc01

    # Build irreversibility
    cfg = IrreversibilityConfig(w_uncert=0.60, w_sparsity=0.25, w_deadend=0.15, deadend_radius=2)
    I_grid = build_irreversibility_map(
        uncertainty_grid=unc_grid,
        feature_density_grid=feat_density,
        free_mask=free_mask,
        cfg=cfg,
    )

    print(f"Unc stats: min={unc_grid.min():.3f} max={unc_grid.max():.3f} mean={unc_grid.mean():.3f}")
    print(f"I stats:   min={I_grid.min():.3f} max={I_grid.max():.3f} mean={I_grid.mean():.3f}")

    # Choose fixed start/goal near a target irreversibility level
    I_target = 0.80
    start, goal, used_band = pick_start_goal_near_I(free_mask, I_grid, I_target=I_target, band=0.05)

    print(f"\nFixed Start: {start}  Fixed Goal: {goal}")
    print(f"I(start)={I_grid[start]:.3f}  I(goal)={I_grid[goal]:.3f}  (target={I_target:.2f}, band=±{used_band:.2f})")

    # tau sweep
    taus = [round(x, 2) for x in np.arange(0.30, 1.01, 0.05)]
    rows = []

    for tau in taus:
        res = astar_irreversibility_constrained(
            free_mask=free_mask,
            irreversibility_grid=I_grid,
            start=start,
            goal=goal,
            tau=tau,
            step_cost=1.0,
        )

        if res.success:
            path_I = [I_grid[y, x] for (y, x) in res.path]
            maxI = float(np.max(path_I))
            meanI = float(np.mean(path_I))
            path_len = int(len(res.path))
        else:
            maxI = np.nan
            meanI = np.nan
            path_len = 0

        rows.append({
            "tau": tau,
            "success": int(res.success),
            "cost": float(res.cost) if np.isfinite(res.cost) else np.inf,
            "path_len": path_len,
            "expansions": int(res.expansions),
            "max_I_on_path": maxI,
            "mean_I_on_path": meanI,
            "I_start": float(I_grid[start]),
            "I_goal": float(I_grid[goal]),
            "reason": res.reason,
        })

        print(f"tau={tau:.2f} success={res.success} expansions={res.expansions} reason={res.reason}")

    out_csv = "irreversibility_tau_sweep.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
