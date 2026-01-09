import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_path(csv_path: str):
    """
    Φορτώνει διαδρομή από CSV.
    Περιμένουμε στήλες: id, x, z
    Θα χρησιμοποιήσουμε:
      X = x
      Y = z  (για 2D επίπεδο)
    """
    df = pd.read_csv(csv_path)
    if not {"x", "z"}.issubset(df.columns):
        raise RuntimeError(f"{csv_path} πρέπει να έχει στήλες x, z")
    pts = df[["x", "z"]].to_numpy()
    return pts


def path_length(points: np.ndarray) -> float:
    """
    Υπολογίζει συνολικό μήκος διαδρομής.
    """
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return float(seg_lengths.sum())


def shortcut_smoothing(points: np.ndarray, iterations: int = 200) -> np.ndarray:
    """
    Απλό shortcut smoothing:
      - επιλέγει τυχαία ζεύγη (i, j)
      - αντικαθιστά το ενδιάμεσο κομμάτι με ευθεία από p_i σε p_j
      *χωρίς* collision checking (υποθέτουμε ελεύθερο χώρο).
    """
    pts = points.copy()
    n = len(pts)
    if n <= 2:
        return pts

    rng = np.random.default_rng(seed=0)

    for _ in range(iterations):
        i, j = sorted(rng.integers(0, n, size=2))
        if j <= i + 1:
            continue
        # shortcut: κρατάμε μέχρι i, και από j και μετά
        pts = np.vstack([pts[: i + 1], pts[j:]])
        n = len(pts)
        if n <= 2:
            break

    return pts


if __name__ == "__main__":
    input_csv = "replan_waypoints.csv"     # βασισμένο στα δικά σου δεδομένα
    output_csv = "replan_waypoints_smooth.csv"

    print(f"[SMOOTH] Loading path from: {input_csv}")
    path = load_path(input_csv)
    L_orig = path_length(path)
    print(f"[SMOOTH] Original path length: {L_orig:.3f}")

    print("[SMOOTH] Running shortcut smoothing...")
    smooth_path = shortcut_smoothing(path, iterations=500)
    L_smooth = path_length(smooth_path)
    print(f"[SMOOTH] Smoothed path length: {L_smooth:.3f}")
    if L_orig > 1e-6:
        reduction = 100.0 * (L_orig - L_smooth) / L_orig
    else:
        reduction = 0.0
    print(f"[SMOOTH] Path length reduction: {reduction:.2f}%")

    # Αποθήκευση νέας διαδρομής — κρατάμε τα ίδια ονόματα x, z
    df_out = pd.DataFrame({"x": smooth_path[:, 0], "z": smooth_path[:, 1]})
    df_out.to_csv(output_csv, index=False)
    print(f"[SMOOTH] Saved smoothed path to: {output_csv}")

    # Plot σύγκρισης
    plt.figure(figsize=(7, 6))
    plt.plot(path[:, 0], path[:, 1], "-o", markersize=2, label="Original path")
    plt.plot(
        smooth_path[:, 0],
        smooth_path[:, 1],
        "-o",
        markersize=3,
        label="Smoothed path",
    )
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Path Smoothing (Shortcut) on Replan Waypoints")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("path_smoothing_result.png", dpi=200)
    plt.show()

    print("[SMOOTH] Visualization saved as path_smoothing_result.png")
