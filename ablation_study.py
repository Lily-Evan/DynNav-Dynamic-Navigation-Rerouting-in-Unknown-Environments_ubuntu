import pandas as pd
import numpy as np

from info_gain_planner import (
    load_coverage_grid,
    load_uncertainty_grid,
    compute_entropy_map,
    load_cluster_centroids,
)
from multiobj_planner import compute_multiobjective_scores


def run_ablation_mode(
    mode_name: str,
    H_map: np.ndarray,
    U_map: np.ndarray,
    centroids_df: pd.DataFrame,
    robot_row: int,
    robot_col: int,
    w_ent: float,
    w_unc: float,
    w_cost: float,
):
    """
    Τρέχει έναν συγκεκριμένο συνδυασμό βαρών
    και επιστρέφει το καλύτερο viewpoint + metrics.
    """
    df_scores = compute_multiobjective_scores(
        H_map,
        U_map,
        centroids_df,
        robot_row=robot_row,
        robot_col=robot_col,
        window_radius=4,
        w_ent=w_ent,
        w_unc=w_unc,
        w_cost=w_cost,
    )

    # Παίρνουμε τον κορυφαίο στόχο με βάση το 'score'
    best = df_scores.iloc[0]

    result = {
        "mode": mode_name,
        "w_ent": w_ent,
        "w_unc": w_unc,
        "w_cost": w_cost,
        "x": best["x"],
        "y": best["y"],
        "entropy_gain": best["entropy_gain"],
        "uncertainty_gain": best["uncertainty_gain"],
        "distance": best["distance"],
        "score": best["score"],
    }
    return result


if __name__ == "__main__":
    coverage_csv = "coverage_grid_with_uncertainty.csv"
    centroids_csv = "cluster_waypoints.csv"
    output_csv = "ablation_results.csv"

    print(f"[ABL] Loading coverage grid: {coverage_csv}")
    prob_grid = load_coverage_grid(coverage_csv)
    H_map = compute_entropy_map(prob_grid)

    print("[ABL] Loading uncertainty grid...")
    U_map = load_uncertainty_grid(coverage_csv)

    print(f"[ABL] Loading centroids: {centroids_csv}")
    centroids_df = load_cluster_centroids(centroids_csv)

    # Θέση ρομπότ (όπως στα άλλα scripts)
    robot_row, robot_col = 0, 0

    results = []

    # 1) Entropy-only mode
    print("\n[ABL] Mode: ENTROPY-ONLY")
    res_ent = run_ablation_mode(
        mode_name="entropy_only",
        H_map=H_map,
        U_map=U_map,
        centroids_df=centroids_df,
        robot_row=robot_row,
        robot_col=robot_col,
        w_ent=1.0,
        w_unc=0.0,
        w_cost=0.2,
    )
    results.append(res_ent)

    # 2) Uncertainty-only mode
    print("\n[ABL] Mode: UNCERTAINTY-ONLY")
    res_unc = run_ablation_mode(
        mode_name="uncertainty_only",
        H_map=H_map,
        U_map=U_map,
        centroids_df=centroids_df,
        robot_row=robot_row,
        robot_col=robot_col,
        w_ent=0.0,
        w_unc=1.0,
        w_cost=0.2,
    )
    results.append(res_unc)

    # 3) Combined entropy + uncertainty
    print("\n[ABL] Mode: COMBINED")
    res_comb = run_ablation_mode(
        mode_name="combined",
        H_map=H_map,
        U_map=U_map,
        centroids_df=centroids_df,
        robot_row=robot_row,
        robot_col=robot_col,
        w_ent=0.5,
        w_unc=0.3,
        w_cost=0.2,
    )
    results.append(res_comb)

    # Συγκεντρωτικά αποτελέσματα
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)

    print("\n[ABL] Ablation study completed.")
    print(f"[ABL] Saved results to: {output_csv}\n")
    print(df_results)
