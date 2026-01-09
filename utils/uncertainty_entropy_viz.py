import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from info_gain_planner import load_coverage_grid, compute_entropy_map

def load_raw_coverage(csv_path="coverage_grid.csv"):
    df = pd.read_csv(csv_path)
    if not {'row', 'col', 'covered'}.issubset(df.columns):
        raise RuntimeError("coverage_grid.csv must have at least row,col,covered")
    return df

def load_clusters(csv_path="cluster_waypoints.csv"):
    df = pd.read_csv(csv_path)
    if not {"x", "y"}.issubset(df.columns):
        raise RuntimeError("cluster_waypoints.csv πρέπει να έχει στήλες x,y")
    return df

def load_ig_goal(csv_path="ig_next_goal.csv"):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("ig_next_goal.csv είναι άδειο")
    r = df.iloc[0]
    return float(r["goal_x"]), float(r["goal_y"])

if __name__ == "__main__":
    coverage_csv = "coverage_grid_with_uncertainty.csv"
    clusters_csv = "cluster_waypoints.csv"
    ig_goal_csv = "ig_next_goal.csv"

    print(f"[VIZ] Φόρτωση δεδομένων από: {coverage_csv}")
    df = load_raw_coverage(coverage_csv)

    max_row = int(df['row'].max())
    max_col = int(df['col'].max())

    # Grid με entropy (μέσω load_coverage_grid + compute_entropy_map)
    prob_grid = load_coverage_grid(coverage_csv)
    H_map = compute_entropy_map(prob_grid)

    # Grid με uncertainty, αν υπάρχει
    has_unc = 'uncertainty' in df.columns
    if has_unc:
        print("[VIZ] Βρέθηκε στήλη 'uncertainty' – δημιουργία uncertainty heatmap")
        U_map = np.zeros((max_row+1, max_col+1), dtype=float)
        for _, r in df.iterrows():
            i = int(r['row'])
            j = int(r['col'])
            u = float(r['uncertainty'])
            U_map[i, j] = u
    else:
        print("[VIZ] ΔΕΝ βρέθηκε 'uncertainty' – U_map = 0")
        U_map = np.zeros((max_row+1, max_col+1), dtype=float)

    clusters = load_clusters(clusters_csv)
    goal_x, goal_y = load_ig_goal(ig_goal_csv)

    # ========== Plot 1: Entropy ==========
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(H_map, origin="lower", cmap="viridis")
    plt.colorbar(label="Entropy")
    plt.scatter(
        clusters["x"], clusters["y"],
        s=10, edgecolors="white", facecolors="none", label="Clusters"
    )
    plt.scatter(
        [goal_x], [goal_y],
        s=80, marker="*", color="red", label="IG goal"
    )
    plt.title("Entropy Map (coverage + VO uncertainty)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right")

    # ========== Plot 2: Uncertainty ==========
    plt.subplot(1, 2, 2)
    plt.imshow(U_map, origin="lower", cmap="magma")
    plt.colorbar(label="VO Uncertainty")
    plt.scatter(
        clusters["x"], clusters["y"],
        s=10, edgecolors="black", facecolors="none", label="Clusters"
    )
    plt.scatter(
        [goal_x], [goal_y],
        s=80, marker="*", color="cyan", label="IG goal"
    )
    plt.title("VO-based Uncertainty Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("uncertainty_entropy_viz.png", dpi=200)
    print("[VIZ] Αποθηκεύτηκε ως uncertainty_entropy_viz.png")
    # plt.show()
