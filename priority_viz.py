import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from info_gain_planner import (
    load_coverage_grid,
    load_uncertainty_grid,
    compute_entropy_map,
    compute_priority_map
)

def load_ig_goal(csv_path="ig_next_goal.csv"):
    df = pd.read_csv(csv_path)
    r = df.iloc[0]
    return float(r["goal_x"]), float(r["goal_y"])


if __name__ == "__main__":
    coverage_csv = "coverage_grid_with_uncertainty.csv"

    print(f"[VIZ] Loading coverage grid: {coverage_csv}")
    prob_grid = load_coverage_grid(coverage_csv)
    H_map = compute_entropy_map(prob_grid)
    U_map = load_uncertainty_grid(coverage_csv)

    # ΣΥΝΔΥΑΣΜΟΣ ENTROPY + UNCERTAINTY
    w_H = 1.0
    w_U = 1.0
    priority_map = compute_priority_map(H_map, U_map,
                                        w_entropy=w_H,
                                        w_uncertainty=w_U)

    # Φόρτωση IG-based goal για overlay
    print("[VIZ] Loading IG goal...")
    goal_x, goal_y = load_ig_goal("ig_next_goal.csv")

    # ===================================================
    # 1. ENTROPY MAP
    # ===================================================
    plt.figure(figsize=(6, 5))
    im1 = plt.imshow(H_map, origin="lower", cmap="viridis")
    plt.scatter([goal_x], [goal_y], s=80, color="red", marker="*", label="IG goal")
    plt.colorbar(im1, label="Entropy (bits)")
    plt.title("Entropy Map")
    plt.xlabel("X (grid col)")
    plt.ylabel("Y (grid row)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("viz_entropy.png", dpi=200)
    plt.show()  #  <-- Προβολή

    # ===================================================
    # 2. UNCERTAINTY MAP
    # ===================================================
    plt.figure(figsize=(6, 5))
    im2 = plt.imshow(U_map, origin="lower", cmap="plasma")
    plt.scatter([goal_x], [goal_y], s=80, color="white", marker="*", label="IG goal")
    plt.colorbar(im2, label="VO Uncertainty")
    plt.title("Uncertainty Map (VO-based)")
    plt.xlabel("X (grid col)")
    plt.ylabel("Y (grid row)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("viz_uncertainty.png", dpi=200)
    plt.show()  # 🔥

    # ===================================================
    # 3. PRIORITY MAP (H + U)
    # ===================================================
    plt.figure(figsize=(6, 5))
    im3 = plt.imshow(priority_map, origin="lower", cmap="inferno")
    plt.scatter([goal_x], [goal_y], s=80, color="cyan", marker="*", label="IG goal")
    plt.colorbar(im3, label="Priority = H + U")
    plt.title("Priority Map (Entropy + Uncertainty)")
    plt.xlabel("X (grid col)")
    plt.ylabel("Y (grid row)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("viz_priority.png", dpi=200)
    plt.show()  # 🔥

    print("\n[VIZ] Completed successfully.")
