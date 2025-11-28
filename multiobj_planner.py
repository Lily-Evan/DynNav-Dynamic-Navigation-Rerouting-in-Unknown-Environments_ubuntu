import math
import numpy as np
import pandas as pd

from info_gain_planner import (
    load_coverage_grid,
    load_uncertainty_grid,
    compute_entropy_map,
    information_gain,
    load_cluster_centroids,
)


def entropy_gain(entropy_map: np.ndarray,
                 center_row: int,
                 center_col: int,
                 window_radius: int = 4) -> float:
    """IG μόνο από entropy."""
    return information_gain(entropy_map, center_row, center_col, window_radius)


def uncertainty_gain(uncertainty_map: np.ndarray,
                     center_row: int,
                     center_col: int,
                     window_radius: int = 4) -> float:
    """IG μόνο από uncertainty (άθροισμα αβεβαιότητας γύρω από viewpoint)."""
    rows, cols = uncertainty_map.shape
    r_min = max(0, center_row - window_radius)
    r_max = min(rows - 1, center_row + window_radius)
    c_min = max(0, center_col - window_radius)
    c_max = min(cols - 1, center_col + window_radius)

    patch = uncertainty_map[r_min:r_max + 1, c_min:c_max + 1]
    return float(patch.sum())


def normalize(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    a_min = arr.min()
    a_max = arr.max()
    if abs(a_max - a_min) < 1e-9:
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min)


def compute_multiobjective_scores(
    entropy_map: np.ndarray,
    uncertainty_map: np.ndarray,
    centroids_df: pd.DataFrame,
    robot_row: int = 0,
    robot_col: int = 0,
    window_radius: int = 4,
    w_ent: float = 0.5,
    w_unc: float = 0.3,
    w_cost: float = 0.2,
):
    """
    Υπολογίζει multi-objective scores για όλα τα candidate viewpoints.

    Objectives:
      - Entropy gain (θέλουμε ΜΕΓΑΛΟ)      -> f1
      - Uncertainty gain (ΜΕΓΑΛΟ)          -> f2
      - Path cost (απόσταση, ΘΕΛΟΥΜΕ ΜΙΚΡΟ)-> f3

    Score:
      score = w_ent * f1_norm + w_unc * f2_norm - w_cost * f3_norm
    """
    ents = []
    uncs = []
    dists = []
    rows = []
    cols = []
    xs = []
    ys = []

    for _, r in centroids_df.iterrows():
        world_x = float(r["x"])
        world_y = float(r["y"])
        c_row = int(round(world_y))
        c_col = int(round(world_x))

        eg = entropy_gain(entropy_map, c_row, c_col, window_radius)
        ug = uncertainty_gain(uncertainty_map, c_row, c_col, window_radius)
        dist = math.sqrt((c_row - robot_row) ** 2 + (c_col - robot_col) ** 2)

        ents.append(eg)
        uncs.append(ug)
        dists.append(dist)
        rows.append(c_row)
        cols.append(c_col)
        xs.append(world_x)
        ys.append(world_y)

    ents = np.array(ents)
    uncs = np.array(uncs)
    dists = np.array(dists)

    # Κανονικοποίηση
    ents_n = normalize(ents)
    uncs_n = normalize(uncs)
    dists_n = normalize(dists)

    scores = w_ent * ents_n + w_unc * uncs_n - w_cost * dists_n

    data = []
    for i in range(len(rows)):
        data.append({
            "row": rows[i],
            "col": cols[i],
            "x": xs[i],
            "y": ys[i],
            "entropy_gain": ents[i],
            "uncertainty_gain": uncs[i],
            "distance": dists[i],
            "entropy_norm": ents_n[i],
            "uncertainty_norm": uncs_n[i],
            "distance_norm": dists_n[i],
            "score": scores[i],
        })

    return pd.DataFrame(data).sort_values("score", ascending=False)


def approximate_pareto_front(df: pd.DataFrame):
    """
    Πολύ απλό Pareto filter:
      - μεγιστοποίηση entropy_gain, uncertainty_gain
      - ελαχιστοποίηση distance

    Κρατάμε σημεία που δεν κυριαρχούνται από άλλα (approx).
    """
    pts = df[["entropy_gain", "uncertainty_gain", "distance"]].to_numpy()
    n = pts.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j κυριαρχεί i αν:
            #  - έχει >= entropy, >= uncertainty
            #  - και <= distance
            if (pts[j, 0] >= pts[i, 0] and
                pts[j, 1] >= pts[i, 1] and
                pts[j, 2] <= pts[i, 2] and
                ((pts[j, 0] > pts[i, 0]) or
                 (pts[j, 1] > pts[i, 1]) or
                 (pts[j, 2] < pts[i, 2]))):
                is_pareto[i] = False
                break

    return df[is_pareto]


if __name__ == "__main__":
    coverage_csv = "coverage_grid_with_uncertainty.csv"
    centroids_csv = "cluster_waypoints.csv"
    output_csv = "multiobj_candidates.csv"

    print(f"[MO] Loading coverage grid: {coverage_csv}")
    prob_grid = load_coverage_grid(coverage_csv)
    H_map = compute_entropy_map(prob_grid)

    print("[MO] Loading uncertainty grid...")
    U_map = load_uncertainty_grid(coverage_csv)

    print(f"[MO] Loading centroids: {centroids_csv}")
    centroids_df = load_cluster_centroids(centroids_csv)

    robot_row, robot_col = 0, 0

    print("[MO] Computing multi-objective scores...")
    df_scores = compute_multiobjective_scores(
        H_map,
        U_map,
        centroids_df,
        robot_row=robot_row,
        robot_col=robot_col,
        window_radius=4,
        w_ent=0.5,
        w_unc=0.3,
        w_cost=0.2,
    )

    df_scores.to_csv(output_csv, index=False)
    print(f"[MO] Saved all candidate scores to: {output_csv}")

    # Top-5 με βάση weighted score
    print("\n[MO] Top-5 candidates by weighted score:")
    print(df_scores.head(5)[["x", "y", "entropy_gain", "uncertainty_gain", "distance", "score"]])

    # Approximate Pareto front
    pareto_df = approximate_pareto_front(df_scores)
    print("\n[MO] Approximate Pareto front (multi-objective non-dominated set):")
    print(pareto_df[["x", "y", "entropy_gain", "uncertainty_gain", "distance", "score"]])

    pareto_df.to_csv("multiobj_pareto_front.csv", index=False)
    print("[MO] Saved Pareto front to: multiobj_pareto_front.csv")
