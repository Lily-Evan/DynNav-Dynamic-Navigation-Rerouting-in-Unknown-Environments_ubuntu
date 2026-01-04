# run_irreversibility_bottleneck_sweep.py
import numpy as np
import pandas as pd

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
)
from irreversibility_planner import astar_irreversibility_constrained


def add_bottleneck_wall(I_grid: np.ndarray, wall_I: float = 0.95, door_I: float = 0.60, thickness: int = 2):
    """
    Create an artificial irreversibility wall in the middle of the grid,
    with a small 'door' gap whose irreversibility is door_I.

    - wall columns: I >= wall_I (almost always blocked unless tau is huge)
    - door patch: I = door_I (opens only when tau >= door_I)
    """
    I2 = I_grid.copy()
    h, w = I2.shape
    mid = w // 2

    t = max(1, int(thickness))
    c0 = max(0, mid - t // 2)
    c1 = min(w, c0 + t)

    # Wall
    I2[:, c0:c1] = np.maximum(I2[:, c0:c1], wall_I)

    # Door (small gap)
    door_row = h // 2
    r0 = max(0, door_row - 1)
    r1 = min(h, door_row + 2)
    I2[r0:r1, c0:c1] = door_I

    return I2, (c0, c1), (r0, r1)


def pick_start_left_goal_right(free_mask: np.ndarray, I_grid: np.ndarray, wall_cols, I_max: float = 0.50):
    """
    Pick start on the LEFT side of the wall and goal on the RIGHT side,
    both with low I (<= I_max), to force crossing the bottleneck.
    """
    c0, c1 = wall_cols
    h, w = free_mask.shape

    # Left region: cols < c0
    left = np.zeros_like(free_mask, dtype=bool)
    left[:, :c0] = True

    # Right region: cols >= c1
    right = np.zeros_like(free_mask, dtype=bool)
    right[:, c1:] = True

    left_candidates = free_mask & left & (I_grid <= I_max)
    right_candidates = free_mask & right & (I_grid <= I_max)

    L = np.argwhere(left_candidates)
    R = np.argwhere(right_candidates)

    if L.shape[0] < 1 or R.shape[0] < 1:
        raise RuntimeError("Could not find start/goal on both sides with low I. Increase I_max.")

    start = tuple(map(int, L[0]))
    goal = tuple(map(int, R[-1]))
    return start, goal


def main():
    path_csv = "coverage_grid_with_uncertainty.csv"

    unc_grid = load_grid_from_cell_table_csv(
        path_csv,
        value_col="uncertainty",
        row_col=("row", "col"),
        fill_nan_with="max",
    )
    free_mask = np.isfinite(unc_grid)

    # proxy feature density
    unc01 = (unc_grid - unc_grid.min()) / (unc_grid.max() - unc_grid.min() + 1e-9)
    feat_density = 1.0 - unc01

    cfg = IrreversibilityConfig(w_uncert=0.60, w_sparsity=0.25, w_deadend=0.15, deadend_radius=2)
    I_grid = build_irreversibility_map(
        uncertainty_grid=unc_grid,
        feature_density_grid=feat_density,
        free_mask=free_mask,
        cfg=cfg,
    )

    # ---- Bottleneck parameters ----
    wall_I = 0.95     # almost always blocked unless tau >= 0.95
    door_I = 0.60     # door opens only when tau >= 0.60
    thickness = 2
    I_bottle, wall_cols, door_rows = add_bottleneck_wall(I_grid, wall_I=wall_I, door_I=door_I, thickness=thickness)

    # Choose start/goal on different sides to FORCE crossing the wall
    start, goal = pick_start_left_goal_right(free_mask, I_bottle, wall_cols=wall_cols, I_max=0.50)

    print(f"Wall cols: {wall_cols}, Door rows: {door_rows}, wall_I={wall_I}, door_I={door_I}")
    print(f"Fixed Start: {start}  Fixed Goal: {goal}")
    print(f"I(start)={I_bottle[start]:.3f}  I(goal)={I_bottle[goal]:.3f}")

    taus = [round(x, 2) for x in np.arange(0.30, 1.01, 0.05)]
    rows = []

    for tau in taus:
        res = astar_irreversibility_constrained(
            free_mask=free_mask,
            irreversibility_grid=I_bottle,
            start=start,
            goal=goal,
            tau=tau,
            step_cost=1.0,
        )

        if res.success:
            path_I = [I_bottle[y, x] for (y, x) in res.path]
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
            "I_start": float(I_bottle[start]),
            "I_goal": float(I_bottle[goal]),
            "reason": res.reason,
        })

        print(f"tau={tau:.2f} success={res.success} expansions={res.expansions} reason={res.reason}")

    out_csv = "irreversibility_bottleneck_tau_sweep.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
