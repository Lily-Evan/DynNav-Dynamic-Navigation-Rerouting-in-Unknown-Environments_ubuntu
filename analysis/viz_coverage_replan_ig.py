import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_coverage_grid(csv_path: str):
    df = pd.read_csv(csv_path)
    required = {"row", "col", "covered"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"{csv_path} πρέπει να έχει στήλες {required}")

    max_row = int(df["row"].max())
    max_col = int(df["col"].max())
    grid = np.zeros((max_row + 1, max_col + 1), dtype=float)

    for _, r in df.iterrows():
        row = int(r["row"])
        col = int(r["col"])
        cov = float(r["covered"])
        grid[row, col] = 1.0 if cov >= 0.5 else 0.0

    return grid


if __name__ == "__main__":
    coverage_csv = "coverage_grid.csv"
    replan_csv = "replan_waypoints.csv"
    clusters_csv = "cluster_waypoints.csv"
    ig_goal_csv = "ig_next_goal.csv"

    print(f"[VIZ] Loading coverage from: {coverage_csv}")
    cov_grid = load_coverage_grid(coverage_csv)

    print(f"[VIZ] Loading replan waypoints from: {replan_csv}")
    df_wp = pd.read_csv(replan_csv)
    if not {"x", "z"}.issubset(df_wp.columns):
        raise RuntimeError(f"{replan_csv} πρέπει να έχει στήλες x, z")
    xs = df_wp["x"].to_numpy()
    zs = df_wp["z"].to_numpy()

    print(f"[VIZ] Loading clusters from: {clusters_csv}")
    df_cl = pd.read_csv(clusters_csv)
    if not {"x", "y"}.issubset(df_cl.columns):
        raise RuntimeError(f"{clusters_csv} πρέπει να έχει στήλες x, y")

    print(f"[VIZ] Loading IG goal from: {ig_goal_csv}")
    df_goal = pd.read_csv(ig_goal_csv)
    gx = float(df_goal.iloc[0]["goal_x"])
    gy = float(df_goal.iloc[0]["goal_y"])

    plt.figure(figsize=(7, 6))
    plt.imshow(cov_grid, origin="lower", cmap="Greys", alpha=0.6)
    plt.colorbar(label="Covered (0/1)")

    # replan path
    plt.plot(xs, zs, "-", linewidth=1.5, label="Replan path")

    # clusters (uncovered regions)
    plt.scatter(df_cl["x"], df_cl["y"], s=15, edgecolors="blue",
                facecolors="none", label="Uncovered clusters")

    # IG-based goal
    plt.scatter([gx], [gy], marker="*", s=180, color="red", label="IG-based goal")

    plt.xlabel("X / col")
    plt.ylabel("Z / row")
    plt.title("Coverage + Replan path + IG goal + uncovered clusters")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("coverage_replan_ig.png", dpi=200)
    plt.show()
    print("[VIZ] Saved coverage_replan_ig.png")
