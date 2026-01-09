import numpy as np
import pandas as pd
import math
import json
import os

from info_gain_planner import (
    load_coverage_grid,
    load_uncertainty_grid,
    compute_entropy_map,
    load_cluster_centroids,
)


# =========================
# 1. IG-like windowed gain
# =========================
def window_sum(grid: np.ndarray, center_row: int, center_col: int, radius: int = 4) -> float:
    rows, cols = grid.shape
    r_min = max(0, center_row - radius)
    r_max = min(rows - 1, center_row + radius)
    c_min = max(0, center_col - radius)
    c_max = min(cols - 1, center_col + radius)
    patch = grid[r_min : r_max + 1, c_min : c_max + 1]
    return float(np.nansum(patch))


# =========================
# 2. VLA config
# =========================
def load_vla_config(path: str = "vla_config.json"):
    """
    Φορτώνει VLA config αν υπάρχει.
    Επιστρέφει:
      {
        "raw_command": ...,
        "region": "full" | "top-left" | "top-right" | "bottom" | "high-uncertainty",
        "priority": "entropy" | "uncertainty" | "combined"
      }
    ή default config αν δεν υπάρχει.
    """
    if not os.path.exists(path):
        print("[MO+VLA] No vla_config.json found. Using default config.")
        return {
            "raw_command": "",
            "region": "full",
            "priority": "combined",
        }
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("[MO+VLA] Loaded VLA config:", cfg)
    # safety defaults
    cfg.setdefault("region", "full")
    cfg.setdefault("priority", "combined")
    return cfg


def vla_to_weights(priority: str):
    """
    Χαρτογραφεί priority → (w_ent, w_unc, w_cost).
    """
    if priority == "entropy":
        w_ent, w_unc, w_cost = 1.0, 0.0, 0.2
    elif priority == "uncertainty":
        w_ent, w_unc, w_cost = 0.0, 1.0, 0.2
    else:  # "combined" ή οτιδήποτε άλλο
        w_ent, w_unc, w_cost = 0.5, 0.3, 0.2
    return w_ent, w_unc, w_cost


def apply_region_filter(centroids_df: pd.DataFrame,
                        region: str,
                        H_map: np.ndarray,
                        U_map: np.ndarray) -> pd.DataFrame:
    """
    Εφαρμόζει region mask στα centroids, ανάλογα με το VLA config.
    - full: επιστρέφει όλα
    - top-left, top-right, bottom: βασισμένο σε x,y bounds
    - high-uncertainty: κρατά κέντρα με υψηλή mean uncertainty γύρω από το σημείο
    """
    if centroids_df.empty:
        return centroids_df

    if region == "full":
        print("[MO+VLA] Region = full → κρατάμε όλα τα clusters.")
        return centroids_df

    xs = centroids_df["x"].to_numpy()
    ys = centroids_df["y"].to_numpy()

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    mask = np.ones(len(centroids_df), dtype=bool)

    if region == "top-left":
        # x μικρό, y μεγάλο (υποθέτουμε origin κάτω-αριστερά)
        mask = (xs <= x_mid) & (ys >= y_mid)
        print("[MO+VLA] Region = top-left")
    elif region == "top-right":
        mask = (xs > x_mid) & (ys >= y_mid)
        print("[MO+VLA] Region = top-right")
    elif region == "bottom":
        mask = (ys <= y_mid)
        print("[MO+VLA] Region = bottom")
    elif region == "high-uncertainty":
        print("[MO+VLA] Region = high-uncertainty (based on U_map).")
        # Για κάθε centroid, μετράμε τοπική mean uncertainty και κρατάμε αυτά πάνω από threshold
        rows_u, cols_u = U_map.shape
        u_means = []
        radius = 3
        for x, y in zip(xs, ys):
            c_row = int(round(y))
            c_col = int(round(x))
            r_min = max(0, c_row - radius)
            r_max = min(rows_u - 1, c_row + radius)
            c_min = max(0, c_col - radius)
            c_max = min(cols_u - 1, c_col + radius)
            patch = U_map[r_min : r_max + 1, c_min : c_max + 1]
            u_means.append(float(np.nanmean(patch)))
        u_means = np.array(u_means)
        # threshold = median των τοπικών μέσων
        thr = np.nanmedian(u_means)
        mask = u_means >= thr
        print(f"[MO+VLA] high-uncertainty threshold (median local U) = {thr:.4f}")
    else:
        print(f"[MO+VLA] Unknown region '{region}', falling back to full.")
        return centroids_df

    filtered = centroids_df[mask].copy()
    print(f"[MO+VLA] Region filter kept {len(filtered)}/{len(centroids_df)} clusters.")
    # Αν τα μηδενίσαμε όλα, fallback στα όλα
    if filtered.empty:
        print("[MO+VLA] WARNING: region filter έδωσε 0 clusters. Χρησιμοποιούμε όλα.")
        return centroids_df
    return filtered


# =========================
# 3. Multi-objective scores
# =========================
def compute_multiobjective_scores(
    H_map: np.ndarray,
    U_map: np.ndarray,
    centroids_df: pd.DataFrame,
    robot_row: int = 0,
    robot_col: int = 0,
    window_radius: int = 4,
    w_ent: float = 0.5,
    w_unc: float = 0.3,
    w_cost: float = 0.2,
) -> pd.DataFrame:

    rows, cols = H_map.shape

    records = []
    for _, r in centroids_df.iterrows():
        cx = float(r["x"])
        cy = float(r["y"])

        c_row = int(round(cy))
        c_col = int(round(cx))

        c_row = max(0, min(rows - 1, c_row))
        c_col = max(0, min(cols - 1, c_col))

        ent_gain = window_sum(H_map, c_row, c_col, window_radius)
        unc_gain = window_sum(U_map, c_row, c_col, window_radius)
        dist = math.sqrt((c_row - robot_row) ** 2 + (c_col - robot_col) ** 2)

        # απλό normalized cost term
        cost_term = dist
        score = w_ent * ent_gain + w_unc * unc_gain - w_cost * cost_term

        records.append(
            {
                "x": cx,
                "y": cy,
                "entropy_gain": ent_gain,
                "uncertainty_gain": unc_gain,
                "distance": dist,
                "score": score,
            }
        )

    df_scores = pd.DataFrame.from_records(records)
    df_scores.sort_values(by="score", ascending=False, inplace=True)
    df_scores.reset_index(drop=True, inplace=True)
    return df_scores


def approximate_pareto_front(df_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Απλό non-dominated filtering στο (entropy_gain, uncertainty_gain, distance).
    Θέλουμε:
      - maximize entropy_gain, uncertainty_gain
      - minimize distance
    """
    if df_scores.empty:
        return df_scores

    pareto_indices = []
    vals = df_scores[["entropy_gain", "uncertainty_gain", "distance"]].to_numpy()

    for i, pi in enumerate(vals):
        dominated = False
        for j, pj in enumerate(vals):
            if j == i:
                continue
            better_or_equal = (
                (pj[0] >= pi[0]) and
                (pj[1] >= pi[1]) and
                (pj[2] <= pi[2])
            )
            strictly_better = (
                (pj[0] > pi[0]) or
                (pj[1] > pi[1]) or
                (pj[2] < pi[2])
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)

    pf = df_scores.iloc[pareto_indices].copy()
    pf.reset_index(drop=True, inplace=True)
    return pf


# =========================
# 4. Main demo with VLA
# =========================
if __name__ == "__main__":
    coverage_csv = "coverage_grid_with_uncertainty.csv"
    centroids_csv = "cluster_waypoints.csv"
    candidates_csv = "multiobj_candidates.csv"
    pareto_csv = "multiobj_pareto_front.csv"

    print(f"[MO] Loading coverage grid: {coverage_csv}")
    prob_grid = load_coverage_grid(coverage_csv)
    H_map = compute_entropy_map(prob_grid)

    print("[MO] Loading uncertainty grid...")
    U_map = load_uncertainty_grid(coverage_csv)

    print(f"[MO] Loading centroids: {centroids_csv}")
    centroids_df = load_cluster_centroids(centroids_csv)

    # === VLA config ===
    vla_cfg = load_vla_config()
    region = vla_cfg["region"]
    priority = vla_cfg["priority"]

    # από priority → weights
    w_ent, w_unc, w_cost = vla_to_weights(priority)
    print(f"[MO+VLA] Using weights from priority='{priority}': "
          f"w_ent={w_ent}, w_unc={w_unc}, w_cost={w_cost}")

    # region filter στα clusters
    centroids_filtered = apply_region_filter(centroids_df, region, H_map, U_map)

    # θέση ρομπότ στο (0,0) σε grid coords
    robot_row, robot_col = 0, 0

    print("[MO] Computing multi-objective scores...")
    df_scores = compute_multiobjective_scores(
        H_map,
        U_map,
        centroids_filtered,
        robot_row=robot_row,
        robot_col=robot_col,
        window_radius=4,
        w_ent=w_ent,
        w_unc=w_unc,
        w_cost=w_cost,
    )
    df_scores.to_csv(candidates_csv, index=False)
    print(f"[MO] Saved all candidate scores to: {candidates_csv}")

    # top-5
    print("\n[MO] Top-5 candidates by weighted score:")
    print(df_scores.head(5))

    # Pareto front
    pf = approximate_pareto_front(df_scores)
    pf.to_csv(pareto_csv, index=False)
    print("\n[MO] Approximate Pareto front (multi-objective non-dominated set):")
    print(pf)
    print(f"[MO] Saved Pareto front to: {pareto_csv}")
