# run_irreversibility_demo.py
import numpy as np

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
    load_covered_mask_from_cell_table_csv,
)
from irreversibility_planner import astar_irreversibility_constrained


def pick_start_goal_safe(free_mask: np.ndarray, I_grid: np.ndarray, tau: float):
    """
    Pick start/goal among cells that are:
      - free
      - irreversibility I <= tau

    We pick two far-apart cells by choosing the first and last safe cell
    in the scan order.
    """
    safe = free_mask & (I_grid <= tau)
    safe_cells = np.argwhere(safe)

    if safe_cells.shape[0] < 2:
        raise RuntimeError(f"Not enough safe cells for tau={tau:.2f}. Try higher tau.")

    start = tuple(map(int, safe_cells[0]))
    goal = tuple(map(int, safe_cells[-1]))
    return start, goal


def main():
    path_csv = "coverage_grid_with_uncertainty.csv"

    # 1) Load uncertainty grid[row,col] from the table CSV
    unc_grid = load_grid_from_cell_table_csv(
        path_csv,
        value_col="uncertainty",
        row_col=("row", "col"),
        fill_nan_with="max",
    )

    # 2) Load covered mask (0/1) from same CSV (not mandatory for movement here)
    covered_mask = load_covered_mask_from_cell_table_csv(
        path_csv,
        covered_col="covered",
        row_col=("row", "col"),
    )

    # 3) Define free space for this demo:
    # Use all finite cells as free. (If you want, AND with covered_mask.)
    free_mask = np.isfinite(unc_grid)

    # Optional (only if covered has many 1s):
    # free_mask = free_mask & covered_mask

    # 4) Feature density proxy (demo): higher density where uncertainty is low
    unc01 = (unc_grid - unc_grid.min()) / (unc_grid.max() - unc_grid.min() + 1e-9)
    feat_density = 1.0 - unc01

    # 5) Build irreversibility map
    cfg = IrreversibilityConfig(w_uncert=0.60, w_sparsity=0.25, w_deadend=0.15, deadend_radius=2)
    I_grid = build_irreversibility_map(
        uncertainty_grid=unc_grid,
        feature_density_grid=feat_density,
        free_mask=free_mask,
        cfg=cfg,
    )

    print(f"Unc stats: min={unc_grid.min():.3f} max={unc_grid.max():.3f} mean={unc_grid.mean():.3f}")
    print(f"I stats:   min={I_grid.min():.3f} max={I_grid.max():.3f} mean={I_grid.mean():.3f}")
    print(f"I(0,0) = {I_grid[0,0]:.3f}")

    # 6) Run constrained A* for different tau values
    taus = [0.99, 0.90, 0.75, 0.65, 0.55]

    for tau in taus:
        # pick start/goal that are feasible under this tau
        try:
            start, goal = pick_start_goal_safe(free_mask, I_grid, tau)
        except RuntimeError as e:
            print(f"\n[tau={tau:.2f}] {e}")
            continue

        print(f"\n=== tau={tau:.2f} ===")
        print(f"Start: {start}  Goal: {goal}")
        print(f"I(start)={I_grid[start]:.3f}  I(goal)={I_grid[goal]:.3f}")

        res = astar_irreversibility_constrained(
            free_mask=free_mask,
            irreversibility_grid=I_grid,
            start=start,
            goal=goal,
            tau=tau,
            step_cost=1.0,
        )

        print(f"success={res.success} cost={res.cost} expansions={res.expansions} reason={res.reason}")

        if res.success:
            path_I = [I_grid[y, x] for (y, x) in res.path]
            print(f"  path length: {len(res.path)}  max I on path: {max(path_I):.3f}  mean I: {np.mean(path_I):.3f}")


if __name__ == "__main__":
    main()

