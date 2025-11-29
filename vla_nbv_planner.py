import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from multiobj_planner import load_vla_config
from info_gain_planner import (
    load_coverage_grid,
    load_uncertainty_grid,
    compute_entropy_map,
    load_cluster_centroids,
)


def generate_viewpoints(x, y, radius=2.0, count=8):
    """
    Δημιουργεί 'count' viewpoints γύρω από το cluster centroid (x,y)
    σε κύκλο.
    """
    pts = []
    for i in range(count):
        angle = 2 * math.pi * (i / count)
        vx = x + radius * math.cos(angle)
        vy = y + radius * math.sin(angle)
        pts.append((vx, vy))
    return pts


def vla_adjust_viewpoint_count(priority):
    """
    Η φυσική γλώσσα αλλάζει πόσα viewpoints δημιουργούμε:
      entropy      -> περισσότερα viewpoints (λεπτομερής αναζήτηση)
      uncertainty  -> λιγότερα viewpoints (focus)
      combined     -> μεσαίο
    """
    if priority == "entropy":
        return 12
    elif priority == "uncertainty":
        return 6
    else:
        return 8


def vla_adjust_radius(region):
    """
    Η φυσική γλώσσα αλλάζει την ακτίνα των viewpoints:
      top-left, top-right -> μεγαλύτερη ακτίνα
      bottom -> μικρή ακτίνα
      full -> default
    """
    if region == "top-left" or region == "top-right":
        return 3.0
    elif region == "bottom":
        return 1.5
    else:
        return 2.0


def score_viewpoint(vx, vy, H_map, U_map, robot_row=0, robot_col=0,
                    w_H=1.0, w_U=1.0, eps=1e-6):
    """
    Υπολογισμός NBV score ενός viewpoint.
    """
    r = int(round(vy))
    c = int(round(vx))

    if r < 0 or r >= H_map.shape[0] or c < 0 or c >= H_map.shape[1]:
        return -9999  # invalid

    H = H_map[r, c]
    U = U_map[r, c]

    dist = math.sqrt((r - robot_row)**2 + (c - robot_col)**2)
    return (w_H * H + w_U * U) / (dist + eps)


if __name__ == "__main__":
    coverage_csv = "coverage_grid_with_uncertainty_vla.csv"
    clusters_csv = "cluster_waypoints.csv"
    output_goal = "vla_nbv_goal.csv"
    output_fig = "nbv_vla_viz.png"

    print("[NBV-VLA] Loading coverage...")
    prob_grid = load_coverage_grid(coverage_csv)
    H_map = compute_entropy_map(prob_grid)
    U_map = load_uncertainty_grid(coverage_csv)

    print("[NBV-VLA] Loading VLA config...")
    cfg = load_vla_config()
    region = cfg["region"]
    priority = cfg["priority"]

    print(f"[NBV-VLA] region={region}, priority={priority}")

    print("[NBV-VLA] Loading clusters...")
    clusters = load_cluster_centroids(clusters_csv)

    # VLA modifies viewpoint generation
    vp_count = vla_adjust_viewpoint_count(priority)
    vp_radius = vla_adjust_radius(region)

    # VLA modifies weighting
    if priority == "entropy":
        w_H, w_U = 1.2, 0.5
    elif priority == "uncertainty":
        w_H, w_U = 0.5, 1.3
    else:
        w_H, w_U = 1.0, 1.0

    best_score = -9999
    best_view = None

    all_viewpoints = []

    # Generate viewpoints around each cluster
    for _, row in clusters.iterrows():
        cx, cy = row["x"], row["y"]
        vps = generate_viewpoints(cx, cy, radius=vp_radius, count=vp_count)

        for vx, vy in vps:
            sc = score_viewpoint(vx, vy, H_map, U_map, 0, 0, w_H, w_U)
            all_viewpoints.append((vx, vy, sc))

            if sc > best_score:
                best_score = sc
                best_view = (vx, vy, sc)

    print("\n[NBV-VLA] === Best NBV Viewpoint ===")
    print(best_view)
    print("===================================\n")

    # Save as CSV
    df_out = pd.DataFrame([{
        "x": best_view[0],
        "y": best_view[1],
        "score": best_view[2],
        "region": region,
        "priority": priority
    }])
    df_out.to_csv(output_goal, index=False)
    print(f"[NBV-VLA] Saved NBV goal: {output_goal}")

    # === Visualization ===
    plt.figure(figsize=(7, 7))
    plt.imshow(H_map, origin="lower", cmap="viridis")
    plt.colorbar(label="Entropy")

    # plot all viewpoints
    xs = [v[0] for v in all_viewpoints]
    ys = [v[1] for v in all_viewpoints]
    plt.scatter(xs, ys, s=20, color="white", alpha=0.35, label="All viewpoints")

    # plot best viewpoint
    plt.scatter(best_view[0], best_view[1], s=200, marker="*", color="red",
                label="Best NBV (VLA)")

    plt.title(f"VLA-driven NBV Viewpoint Selection\nregion={region}, priority={priority}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_fig, dpi=220)
    plt.show()
    print(f"[NBV-VLA] Saved visualization: {output_fig}")
