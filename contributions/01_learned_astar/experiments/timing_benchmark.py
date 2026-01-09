import time
import pandas as pd

from info_gain_planner import (
    load_coverage_grid,
    load_uncertainty_grid,
    compute_entropy_map,
    load_cluster_centroids,
)
from multiobj_planner import (
    compute_multiobjective_scores,
    approximate_pareto_front,
)

N_RUNS = 50  # πόσες φορές θα τρέξει κάθε μέθοδος για μέσο χρόνο


def benchmark_multiobj(H_map, U_map, centroids_df, robot_row=0, robot_col=0):
    """
    Μετράει χρόνο εκτέλεσης του multi-objective planner.
    """
    start = time.perf_counter()
    df_scores = None
    for _ in range(N_RUNS):
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
    end = time.perf_counter()
    avg_time = (end - start) / N_RUNS
    return avg_time, df_scores


def benchmark_pareto(df_scores):
    """
    Μετράει χρόνο υπολογισμού του Pareto front.
    """
    start = time.perf_counter()
    for _ in range(N_RUNS):
        _ = approximate_pareto_front(df_scores)
    end = time.perf_counter()
    avg_time = (end - start) / N_RUNS
    return avg_time


if __name__ == "__main__":
    coverage_csv = "coverage_grid_with_uncertainty.csv"
    centroids_csv = "cluster_waypoints.csv"

    print(f"[TIMING] Loading coverage grid: {coverage_csv}")
    prob_grid = load_coverage_grid(coverage_csv)
    H_map = compute_entropy_map(prob_grid)

    print("[TIMING] Loading uncertainty grid...")
    U_map = load_uncertainty_grid(coverage_csv)

    print(f"[TIMING] Loading centroids: {centroids_csv}")
    centroids_df = load_cluster_centroids(centroids_csv)

    robot_row, robot_col = 0, 0

    print("\n[TIMING] Benchmarking multi-objective planner...")
    t_mo, df_scores = benchmark_multiobj(H_map, U_map, centroids_df, robot_row, robot_col)
    print(f"[TIMING] Multi-objective planner avg time over {N_RUNS} runs: {t_mo*1000:.3f} ms")

    print("\n[TIMING] Benchmarking Pareto front computation...")
    t_pf = benchmark_pareto(df_scores)
    print(f"[TIMING] Pareto front avg time over {N_RUNS} runs: {t_pf*1000:.3f} ms")

    # Αποθήκευση αποτελεσμάτων σε CSV
    results = pd.DataFrame(
        [
            {"component": "Multiobjective_planner", "avg_time_ms": t_mo * 1000.0},
            {"component": "Pareto_front", "avg_time_ms": t_pf * 1000.0},
        ]
    )
    results.to_csv("timing_results.csv", index=False)
    print("\n[TIMING] Saved timing_results.csv")
    print(results)
