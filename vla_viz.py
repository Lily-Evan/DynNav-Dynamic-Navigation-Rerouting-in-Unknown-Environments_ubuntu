import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from info_gain_planner import (
    load_coverage_grid,
    load_uncertainty_grid,
    compute_entropy_map,
    load_cluster_centroids,
)
from multiobj_planner import (
    load_vla_config,
    vla_to_weights,
    apply_region_filter,
    compute_multiobjective_scores,
)


def visualize_vla_run():
    # === Load maps ===
    coverage_csv = "coverage_grid_with_uncertainty.csv"
    centroids_csv = "cluster_waypoints.csv"

    print("[VLA-VIZ] Loading maps...")
    prob_grid = load_coverage_grid(coverage_csv)
    H_map = compute_entropy_map(prob_grid)
    U_map = load_uncertainty_grid(coverage_csv)
    centroids_df = load_cluster_centroids(centroids_csv)

    # === Load VLA config ===
    cfg = load_vla_config()
    region = cfg["region"]
    priority = cfg["priority"]

    w_ent, w_unc, w_cost = vla_to_weights(priority)
    print(f"[VLA-VIZ] Using weights: w_ent={w_ent}, w_unc={w_unc}, w_cost={w_cost}")
    print(f"[VLA-VIZ] Region filter: {region}")

    # === Filter clusters ===
    filtered = apply_region_filter(centroids_df, region, H_map, U_map)

    # === Compute scores for filtered region ===
    df_scores = compute_multiobjective_scores(
        H_map,
        U_map,
        filtered,
        robot_row=0,
        robot_col=0,
        window_radius=4,
        w_ent=w_ent,
        w_unc=w_unc,
        w_cost=w_cost,
    )

    # === Pick best candidate ===
    best = df_scores.iloc[0]
    best_x = best["x"]
    best_y = best["y"]

    # === Visualization ===
    plt.figure(figsize=(7, 7))
    plt.imshow(H_map, origin="lower", cmap="viridis")
    plt.colorbar(label="Entropy")

    # Plot ALL clusters in light color
    plt.scatter(
        centroids_df["x"],
        centroids_df["y"],
        s=20,
        edgecolors="white",
        facecolors="none",
        alpha=0.2,
        label="All clusters",
    )

    # Plot FILTERED clusters (those selected by VLA region)
    plt.scatter(
        filtered["x"],
        filtered["y"],
        s=40,
        edgecolors="yellow",
        facecolors="none",
        linewidths=1.5,
        label=f"VLA region: {region}",
    )

    # Plot best goal
    plt.scatter(
        [best_x],
        [best_y],
        s=120,
        marker="*",
        color="red",
        label="VLA-selected goal",
    )

    plt.title("VLA-Controlled Exploration Result")
    plt.xlabel("X (grid col)")
    plt.ylabel("Y (grid row)")
    plt.legend(loc="upper right")

    out_path = "vla_run_viz.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    print(f"[VLA-VIZ] Saved visualization to: {out_path}")


if __name__ == "__main__":
    visualize_vla_run()
