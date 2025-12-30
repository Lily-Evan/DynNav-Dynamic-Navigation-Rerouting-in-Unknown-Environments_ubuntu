# ================================================================
# Uncertainty Calibration Evaluation
#
# Φορτώνει:
#   - predicted means
#   - predicted variances
#   - ground truth y
#
# Υπολογίζει:
#   - MSE
#   - NLL (Gaussian)
#   - UCE (Uncertainty Calibration Error)
#
# Παράγει:
#   - calibration summary CSV
#   - reliability plot (std_pred vs sqrt(MSE per bin))
#   - histogram λάθους / variance
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gaussian_nll(y, mu, var, eps=1e-6):
    var_clamped = np.maximum(var, eps)
    return 0.5 * (np.log(2 * np.pi * var_clamped) + (y - mu) ** 2 / var_clamped)


def compute_uce(error_sq, var, n_bins=10):
    """
    Uncertainty Calibration Error (UCE):
    - κάνουμε binning στην predicted variance
    - σε κάθε bin συγκρίνουμε mean(error_sq) με mean(var)
    - UCE = sum (n_bin/N * |mse_bin - var_bin|)
    """
    var = var.reshape(-1)
    error_sq = error_sq.reshape(-1)

    N = len(var)
    if N == 0:
        return 0.0

    v_min, v_max = float(var.min()), float(var.max())
    if v_min == v_max:
        # όλα ίδια variance → δεν έχει νόημα binning
        return float(np.abs(error_sq.mean() - var.mean()))

    bins = np.linspace(v_min, v_max, n_bins + 1)
    uce = 0.0

    for b in range(n_bins):
        left = bins[b]
        right = bins[b + 1]
        mask = (var >= left) & (var < right) if b < n_bins - 1 else (var >= left) & (var <= right)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        mse_bin = float(error_sq[idx].mean())
        var_bin = float(var[idx].mean())
        weight = len(idx) / N
        uce += weight * abs(mse_bin - var_bin)

    return float(uce)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, "logs_calibration")

    means = np.load(os.path.join(log_dir, "means.npy"))
    vars_ = np.load(os.path.join(log_dir, "vars.npy"))
    y_true = np.load(os.path.join(log_dir, "y_true.npy"))

    print(f"[CAL-EVAL] Loaded means/vars/y from {log_dir}")
    print(f"[CAL-EVAL] N = {len(y_true)}")

    error_sq = (means - y_true) ** 2
    mse = float(error_sq.mean())

    nll_arr = gaussian_nll(y_true, means, vars_)
    nll = float(nll_arr.mean())

    uce = compute_uce(error_sq, vars_, n_bins=10)

    print("\n[CAL-EVAL] Metrics:")
    print(f"  MSE  = {mse:.6f}")
    print(f"  NLL  = {nll:.6f}")
    print(f"  UCE  = {uce:.6f}")

    # ========= Save summary CSV =========
    summary = {
        "mse": [mse],
        "nll": [nll],
        "uce": [uce],
        "num_samples": [int(len(y_true))],
    }
    df_summary = pd.DataFrame(summary)
    summary_path = os.path.join(log_dir, "uncertainty_calibration_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"[CAL-EVAL] Saved summary CSV to: {summary_path}")

    # ========= Reliability Plot =========
    # binning σε variance, plot: sqrt(var_bin) vs sqrt(error_bin)
    n_bins = 10
    v_min, v_max = float(vars_.min()), float(vars_.max())
    if v_min == v_max:
        print("[CAL-EVAL] Skipping reliability plot (variance constant).")
        return

    bins = np.linspace(v_min, v_max, n_bins + 1)
    bin_centers = []
    std_pred_list = []
    std_err_list = []

    for b in range(n_bins):
        left = bins[b]
        right = bins[b + 1]
        mask = (vars_ >= left) & (vars_ < right) if b < n_bins - 1 else (vars_ >= left) & (vars_ <= right)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        var_bin = float(vars_[idx].mean())
        err_bin = float(error_sq[idx].mean())
        bin_centers.append((left + right) * 0.5)
        std_pred_list.append(np.sqrt(var_bin))
        std_err_list.append(np.sqrt(err_bin))

    bin_centers = np.array(bin_centers)
    std_pred_arr = np.array(std_pred_list)
    std_err_arr = np.array(std_err_list)

    # Plot 1: reliability (std_pred vs std_err)
    plt.figure(figsize=(6, 6))
    plt.plot(std_pred_arr, std_err_arr, marker="o", label="per-bin")
    max_val = max(std_pred_arr.max(), std_err_arr.max()) * 1.05
    plt.plot([0, max_val], [0, max_val], "k--", label="perfect calibration")
    plt.xlabel("Predicted Std (sqrt(variance))")
    plt.ylabel("Empirical Std (sqrt(MSE per bin))")
    plt.title("Uncertainty Reliability Diagram")
    plt.grid(True)
    plt.legend()
    rel_path = os.path.join(log_dir, "uncertainty_reliability_plot.png")
    plt.savefig(rel_path, dpi=200)
    plt.close()
    print(f"[CAL-EVAL] Saved reliability plot to: {rel_path}")

    # Plot 2: histogram of error vs variance
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(error_sq, bins=20)
    plt.title("Squared Error Distribution")
    plt.xlabel("Squared Error")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(vars_, bins=20)
    plt.title("Predicted Variance Distribution")
    plt.xlabel("Variance")
    plt.ylabel("Count")

    plt.tight_layout()
    hist_path = os.path.join(log_dir, "uncertainty_error_var_hist.png")
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"[CAL-EVAL] Saved error/var histograms to: {hist_path}")

    print("\n[CAL-EVAL] Calibration evaluation finished successfully 🎯")


if __name__ == "__main__":
    main()
