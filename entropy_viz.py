import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from info_gain_planner import load_coverage_grid, compute_entropy_map

def load_clusters(csv_path="cluster_waypoints.csv"):
    df = pd.read_csv(csv_path)
    if not {"x", "y"}.issubset(df.columns):
        raise RuntimeError(f"cluster_waypoints.csv πρέπει να έχει στήλες x,y")
    return df

def load_ig_goal(csv_path="ig_next_goal.csv"):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("ig_next_goal.csv είναι άδειο")
    r = df.iloc[0]
    return float(r["goal_x"]), float(r["goal_y"])

if __name__ == "__main__":
    # Χρησιμοποιούμε ΤΟ ΙΔΙΟ coverage CSV με τον info_gain_planner,
    # ώστε ο entropy χάρτης να είναι συνεπής με τον IG-based στόχο.
    coverage_csv = "coverage_grid_with_uncertainty.csv"
    clusters_csv = "cluster_waypoints.csv"
    ig_goal_csv = "ig_next_goal.csv"

    print(f"[VIZ] Φόρτωση coverage από: {coverage_csv}")
    # load_coverage_grid δίνει p_unknown(i,j) και compute_entropy_map → H(i,j) σε bits
    prob_grid = load_coverage_grid(coverage_csv)
    H_map = compute_entropy_map(prob_grid)

    print(f"[VIZ] Φόρτωση clusters από: {clusters_csv}")
    clusters = load_clusters(clusters_csv)

    print(f"[VIZ] Φόρτωση IG goal από: {ig_goal_csv}")
    goal_x, goal_y = load_ig_goal(ig_goal_csv)

    # Plot entropy heatmap
    plt.figure(figsize=(6, 6))
    im = plt.imshow(H_map, origin="lower", cmap="viridis")
    cbar = plt.colorbar(im, label="Entropy (bits of uncertainty)")

    # Overlay clusters (x,y) → (col,row)
    plt.scatter(
        clusters["x"], clusters["y"],
        s=10, edgecolors="white", facecolors="none", label="Uncovered clusters"
    )

    # Overlay IG-based goal
    plt.scatter(
        [goal_x], [goal_y],
        s=80, marker="*", color="red", label="IG-based goal"
    )

    plt.title("Entropy Map με Clusters + IG-based Goal")
    plt.xlabel("X (grid col)")
    plt.ylabel("Y (grid row)")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("entropy_ig_viz.png", dpi=200)
    print("[VIZ] Αποθηκεύτηκε ως entropy_ig_viz.png")
    # plt.show()
