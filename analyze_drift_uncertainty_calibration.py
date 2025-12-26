import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from train_drift_uncertainty_net import (
    CSV_PATH,
    MODEL_PATH,
    NORM_STATS_PATH,
    UncertaintyMLP,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_stats():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    input_dim = ckpt["input_dim"]
    feature_cols = ckpt["feature_cols"]
    target_col = ckpt["target_col"]

    model = UncertaintyMLP(input_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    stats = np.load(NORM_STATS_PATH)
    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    return model, feature_cols, target_col, x_mean, x_std, y_mean, y_std


def main():
    model, feature_cols, target_col, x_mean, x_std, y_mean, y_std = load_model_and_stats()

    df = pd.read_csv(CSV_PATH)
    num_df = df.select_dtypes(include=[np.number])

    if target_col not in num_df.columns:
        raise ValueError(f"Target column {target_col} δεν βρέθηκε στο CSV.")

    X = num_df[feature_cols].to_numpy(dtype=np.float32)
    y = num_df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    # normalize inputs
    Xn = (X - x_mean) / x_std

    # forward pass
    with torch.no_grad():
        xb = torch.from_numpy(Xn).to(DEVICE)
        mean_n, log_std = model(xb)
        mean_n = mean_n.cpu().numpy()
        log_std = log_std.cpu().numpy()

    # unnormalize στο original scale του target
    mean_orig = mean_n * y_std + y_mean
    std_norm = np.exp(log_std)
    std_orig = std_norm * y_std  # scale std με y_std

    y = y.reshape(-1, 1)
    err = y - mean_orig           # signed error
    abs_err = np.abs(err)

    # z-score style: |error| / sigma
    z = abs_err / (std_orig + 1e-8)

    # -----------------------------
    # 1) Histogram του z
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(z, bins=30, density=True)
    plt.xlabel("|error| / sigma_pred")
    plt.ylabel("density")
    plt.title("Drift uncertainty calibration: z = |err| / sigma")
    plt.tight_layout()
    plt.savefig("drift_uncertainty_z_hist.png", dpi=200)
    print("[INFO] Saved histogram to drift_uncertainty_z_hist.png")

    # -----------------------------
    # 2) Sigma calibration curve
    # -----------------------------
    sigma = std_orig.flatten()
    z_flat = z.flatten()

    mask = np.isfinite(sigma) & np.isfinite(z_flat)
    sigma = sigma[mask]
    z_flat = z_flat[mask]

    num_bins = 10
    bin_edges = np.linspace(sigma.min(), sigma.max(), num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    avg_z = []
    counts = []

    for i in range(num_bins):
        m = (sigma >= bin_edges[i]) & (sigma < bin_edges[i + 1])
        if m.sum() == 0:
            avg_z.append(np.nan)
            counts.append(0)
        else:
            avg_z.append(z_flat[m].mean())
            counts.append(int(m.sum()))

    avg_z = np.array(avg_z)

    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, avg_z, marker="o")
    plt.axhline(1.0, color="gray", linestyle="--", label="ideal (E[z]=1)")
    plt.xlabel("predicted sigma")
    plt.ylabel("E[|err| / sigma]")
    plt.title("Sigma calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("drift_uncertainty_sigma_calibration.png", dpi=200)
    print("[INFO] Saved calibration curve to drift_uncertainty_sigma_calibration.png")

    # -----------------------------
    # 3) Gaussian-style coverage
    # -----------------------------
    # Ιδανικά (Gaussian):
    # 1σ -> 68.27%, 2σ -> 95.45%, 3σ -> 99.73%
    coverage_levels = [1.0, 2.0, 3.0]
    ideal_cov = [0.6827, 0.9545, 0.9973]

    empirical_cov = []
    for k in coverage_levels:
        cov_k = float((abs_err <= k * std_orig).mean())
        empirical_cov.append(cov_k)

    empirical_cov = np.array(empirical_cov)
    ideal_cov = np.array(ideal_cov)

    cov_diff = empirical_cov - ideal_cov
    abs_cov_diff = np.abs(cov_diff)
    gaussian_ece = float(abs_cov_diff.mean())

    # plot coverage bars
    x = np.arange(len(coverage_levels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, ideal_cov, width, label="ideal Gaussian")
    plt.bar(x + width/2, empirical_cov, width, label="empirical")
    plt.xticks(x, [f"{k}σ" for k in coverage_levels])
    plt.ylim(0.0, 1.05)
    plt.ylabel("coverage probability")
    plt.title("Gaussian coverage calibration (1σ / 2σ / 3σ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("drift_uncertainty_gaussian_coverage.png", dpi=200)
    print("[INFO] Saved Gaussian coverage figure to drift_uncertainty_gaussian_coverage.png")

    # -----------------------------
    # 4) Εκτύπωση & CSV stats
    # -----------------------------
    print("\n[STATS]")
    print("mean(|err|):", float(abs_err.mean()))
    print("mean(sigma):", float(sigma.mean()))
    print("mean(z):", float(z_flat.mean()))
    print("median(z):", float(np.median(z_flat)))
    for k, emp, ideal in zip(coverage_levels, empirical_cov, ideal_cov):
        print(f"coverage |err| <= {k}σ: empirical={emp:.4f}, ideal={ideal:.4f}, diff={emp-ideal:.4f}")
    print("Gaussian-style ECE (mean |empirical-ideal|):", gaussian_ece)

    # Αποθήκευση σε CSV
    rows = []
    for k, emp, ideal, diff in zip(coverage_levels, empirical_cov, ideal_cov, cov_diff):
        rows.append(
            {
                "k_sigma": k,
                "empirical_coverage": emp,
                "ideal_coverage": ideal,
                "coverage_diff": diff,
                "abs_coverage_diff": abs(cov_diff),
            }
        )

    summary = {
        "mean_abs_err": float(abs_err.mean()),
        "mean_sigma": float(sigma.mean()),
        "mean_z": float(z_flat.mean()),
        "median_z": float(np.median(z_flat)),
        "gaussian_ece": gaussian_ece,
    }

    cov_df = pd.DataFrame(rows)
    cov_df.to_csv("drift_uncertainty_gaussian_coverage_stats.csv", index=False)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("drift_uncertainty_summary_stats.csv", index=False)

    print("[INFO] Saved coverage stats to drift_uncertainty_gaussian_coverage_stats.csv")
    print("[INFO] Saved summary stats to drift_uncertainty_summary_stats.csv")


if __name__ == "__main__":
    main()
