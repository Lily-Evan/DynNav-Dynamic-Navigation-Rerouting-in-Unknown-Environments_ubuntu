import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def moving_average(x: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Απλό smoothing με moving average (πρέπει window να είναι περιττός).
    """
    if window < 3:
        return x.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


if __name__ == "__main__":
    vo_csv = "vo_trajectory.csv"
    coverage_csv = "coverage_grid_with_uncertainty.csv"

    print(f"[DRIFT] Loading VO trajectory from: {vo_csv}")
    df_vo = pd.read_csv(vo_csv)
    # περιμένουμε στήλες: frame_idx, x, y, z
    if not {"x", "z"}.issubset(df_vo.columns):
        raise RuntimeError(f"{vo_csv} πρέπει να έχει στήλες x, z")

    xs = df_vo["x"].to_numpy()
    zs = df_vo["z"].to_numpy()

    # 1. Υπολογισμός "smooth" τροχιάς ως αναφορά
    xs_smooth = moving_average(xs, window=21)
    zs_smooth = moving_average(zs, window=21)

    # 2. Drift = απόσταση από smoothed τροχιά (ανά frame)
    drift = np.sqrt((xs - xs_smooth) ** 2 + (zs - zs_smooth) ** 2)

    print(f"[DRIFT] Loading coverage grid (with uncertainty) from: {coverage_csv}")
    df_cov = pd.read_csv(coverage_csv)
    if not {"row", "col", "uncertainty"}.issubset(df_cov.columns):
        raise RuntimeError(f"{coverage_csv} πρέπει να έχει στήλες row, col, uncertainty")

    max_row = int(df_cov["row"].max())
    max_col = int(df_cov["col"].max())

    # Φτιάχνουμε 2D grid αβεβαιότητας για πιο εύκολο indexing
    U_grid = np.zeros((max_row + 1, max_col + 1), dtype=float)
    U_grid[:] = np.nan
    for _, r in df_cov.iterrows():
        row = int(r["row"])
        col = int(r["col"])
        u = float(r["uncertainty"])
        U_grid[row, col] = u

    # 3. Καλύτερο mapping VO → grid:
    #    Κλιμακώνουμε x,z στο [0, max_col] / [0, max_row]
    x_min, x_max = xs.min(), xs.max()
    z_min, z_max = zs.min(), zs.max()

    # Προστασία μηδενικού εύρους
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0
    if abs(z_max - z_min) < 1e-9:
        z_max = z_min + 1.0

    cols = np.round((xs - x_min) / (x_max - x_min) * max_col).astype(int)
    rows = np.round((zs - z_min) / (z_max - z_min) * max_row).astype(int)

    # Clip στα όρια του grid
    rows = np.clip(rows, 0, max_row)
    cols = np.clip(cols, 0, max_col)

    # 4. Συλλογή uncertainty για κάθε frame
    uncert_list = []
    for rr, cc in zip(rows, cols):
        u = U_grid[rr, cc]
        uncert_list.append(u)

    uncert = np.array(uncert_list, dtype=float)

    # 5. Φιλτράρουμε frames όπου έχουμε ορισμένη uncertainty (όχι NaN)
    mask_valid = ~np.isnan(uncert)
    drift_valid = drift[mask_valid]
    uncert_valid = uncert[mask_valid]

    print(f"[DRIFT] Valid samples with mapped uncertainty: {len(drift_valid)}")

    if len(drift_valid) < 5:
        print("[DRIFT] Πολύ λίγα valid samples για ουσιαστική στατιστική ανάλυση.")
    else:
        # 6. Έλεγχος αν υπάρχει διασπορά στην uncertainty
        std_u = float(np.std(uncert_valid))
        can_corr = std_u > 1e-6 and np.std(drift_valid) > 1e-6

        if can_corr:
            corr = np.corrcoef(drift_valid, uncert_valid)[0, 1]
        else:
            corr = np.nan

        # 7. Χωρίζουμε σε bins (low / mid / high) με quantiles
        q1 = np.quantile(uncert_valid, 1.0 / 3.0)
        q2 = np.quantile(uncert_valid, 2.0 / 3.0)

        low_mask = uncert_valid <= q1
        mid_mask = (uncert_valid > q1) & (uncert_valid <= q2)
        high_mask = uncert_valid > q2

        def safe_mean(x):
            return float(x.mean()) if len(x) > 0 else float("nan")

        mean_drift_low = safe_mean(drift_valid[low_mask])
        mean_drift_mid = safe_mean(drift_valid[mid_mask])
        mean_drift_high = safe_mean(drift_valid[high_mask])

        print("\n[DRIFT] === Drift vs Uncertainty Analysis (v2) ===")
        print(f"[DRIFT] Std(uncertainty): {std_u:.4f}")
        print(f"[DRIFT] Pearson correlation (drift, uncertainty): {corr:.4f}")
        print(f"[DRIFT] Q1, Q2 quantiles of uncertainty: {q1:.4f}, {q2:.4f}")
        print(f"[DRIFT] Mean drift in LOW-uncertainty bin:   {mean_drift_low:.4f}")
        print(f"[DRIFT] Mean drift in MID-uncertainty bin:   {mean_drift_mid:.4f}")
        print(f"[DRIFT] Mean drift in HIGH-uncertainty bin:  {mean_drift_high:.4f}")

        # 8. Scatter plot: uncertainty vs drift
        plt.figure(figsize=(7, 6))
        plt.scatter(uncert_valid, drift_valid, s=8, alpha=0.5)
        plt.xlabel("Uncertainty (cell)")
        plt.ylabel("Drift magnitude (VO vs smoothed VO)")
        plt.title("Drift vs Uncertainty (VO projected on coverage grid)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("drift_vs_uncertainty_scatter.png", dpi=200)
        plt.show()
        print("[DRIFT] Saved scatter plot to: drift_vs_uncertainty_scatter.png")

        # 9. Bar plot: mean drift per uncertainty bin
        labels = ["Low U", "Mid U", "High U"]
        means = [mean_drift_low, mean_drift_mid, mean_drift_high]

        plt.figure(figsize=(6, 5))
        plt.bar(labels, means)
        plt.ylabel("Mean drift")
        plt.title("Mean drift per uncertainty bin")
        plt.tight_layout()
        plt.savefig("drift_vs_uncertainty_bins.png", dpi=200)
        plt.show()
        print("[DRIFT] Saved bar plot to: drift_vs_uncertainty_bins.png")
