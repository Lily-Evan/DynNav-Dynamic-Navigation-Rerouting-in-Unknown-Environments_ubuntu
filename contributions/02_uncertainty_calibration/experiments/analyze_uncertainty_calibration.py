import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
ERR_CSV = BASE / "phd_experiments" / "uncertainty_error_data.csv"
OUT_MD = BASE / "phd_experiments" / "uncertainty_calibration_summary.md"
PLOTS = BASE / "phd_experiments" / "uncertainty_plots"


def load_data():
    rows = []
    with ERR_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "instance_id": int(r["instance_id"]),
                "x": int(r["state_x"]),
                "y": int(r["state_y"]),
                "h_star": float(r["h_star"]),
                "mu": float(r["mu"]),
                "sigma": float(r["sigma"]),
                "error": float(r["error"]),
                "abs_error": float(r["abs_error"]),
            })
    return rows


def scatter_sigma_abs_error(sigmas, abs_errors, out_path):
    plt.figure()
    plt.scatter(sigmas, abs_errors, s=5, alpha=0.3)
    plt.xlabel("sigma (predictive std)")
    plt.ylabel("|error| = |mu - h*|")
    plt.title("Abs error vs predictive sigma")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def binned_calibration(sigmas, abs_errors, num_bins=10):
    """Χωρίζουμε τα sigmas σε bins και υπολογίζουμε mean abs_error ανά bin."""
    sigmas = np.array(sigmas)
    abs_errors = np.array(abs_errors)

    # κόβουμε τα extreme outliers για να έχει νόημα το plot
    finite_mask = np.isfinite(sigmas) & np.isfinite(abs_errors)
    sigmas = sigmas[finite_mask]
    abs_errors = abs_errors[finite_mask]

    if len(sigmas) == 0:
        return [], [], []

    bins = np.linspace(sigmas.min(), sigmas.max(), num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_abs = []
    counts = []

    for i in range(num_bins):
        mask = (sigmas >= bins[i]) & (sigmas < bins[i+1])
        if mask.sum() == 0:
            mean_abs.append(float("nan"))
            counts.append(0)
        else:
            mean_abs.append(float(abs_errors[mask].mean()))
            counts.append(int(mask.sum()))

    return bin_centers, mean_abs, counts


def plot_binned_calibration(bin_centers, mean_abs, out_path):
    plt.figure()
    plt.plot(bin_centers, mean_abs, marker="o")
    plt.xlabel("sigma bin center")
    plt.ylabel("mean |error|")
    plt.title("Binned calibration: mean |error| vs predicted sigma")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    if not ERR_CSV.exists():
        print(f"[ERROR] {ERR_CSV} not found. Τρέξε πρώτα run_uncertainty_error_logging.")
        return

    PLOTS.mkdir(parents=True, exist_ok=True)
    data = load_data()

    sigmas = [r["sigma"] for r in data]
    abs_errors = [r["abs_error"] for r in data]

    # 1) scatter |error| vs sigma
    scatter_sigma_abs_error(sigmas, abs_errors, PLOTS / "sigma_vs_abs_error.png")

    # 2) binned calibration
    bin_centers, mean_abs, counts = binned_calibration(sigmas, abs_errors, num_bins=10)
    if bin_centers:
        plot_binned_calibration(bin_centers, mean_abs, PLOTS / "binned_calibration.png")

    # 3) summary markdown
    # απλό summary για αρχή
    lines = []
    lines.append("# Uncertainty Calibration Analysis\n")
    lines.append(f"- Total samples: {len(data)}")
    valid_abs = [e for e in abs_errors if math.isfinite(e)]
    if valid_abs:
        lines.append(f"- Mean absolute error: {sum(valid_abs)/len(valid_abs):.4f}")
    lines.append("\n## Figures\n")
    lines.append("1. Scatter: |error| vs sigma  → `uncertainty_plots/sigma_vs_abs_error.png`")
    lines.append("2. Binned calibration plot   → `uncertainty_plots/binned_calibration.png`")

    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Wrote uncertainty calibration summary to {OUT_MD}")


if __name__ == "__main__":
    main()
