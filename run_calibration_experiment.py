# run_calibration_experiment.py

from pathlib import Path
import ast
import numpy as np
import pandas as pd

from neural_uncertainty.calibration_methods import (
    AffineCalibrator,
    HistogramCalibrator,
    compute_calibration_metrics,
)


def load_calibration_data(csv_path: Path):
    """
    Φορτώνει δεδομένα calibration από CSV.

    Το drift_uncertainty_gaussian_coverage_stats.csv περιέχει στήλες όπου
    κάποιες τιμές είναι stringified arrays, π.χ. "[0.03 0.04 0.002]".
    Τις κάνουμε λίστες και παίρνουμε τον μέσο όρο.
    """
    df = pd.read_csv(csv_path)

    def parse_column(col):
        # Αν η στήλη είναι object (strings), προσπαθούμε να κάνουμε parse arrays
        if col.dtype == object:
            def parse_value(x):
                # Αν είναι NaN, το χειριζόμαστε
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return 0.0

                if isinstance(x, str):
                    s = x.strip()
                    # Περίπτωση: "[0.03 0.04 0.002]"
                    if s.startswith("[") and s.endswith("]"):
                        inner = s[1:-1].strip()
                        if not inner:
                            return 0.0
                        parts = inner.split()
                        vals = [float(p) for p in parts]
                        return float(np.mean(vals))
                    else:
                        # Απλό scalar σε string
                        return float(s)
                else:
                    # Ήδη αριθμός
                    return float(x)

            return col.apply(parse_value).to_numpy()
        else:
            return col.to_numpy(dtype=float)

    # Εδώ διαλέγουμε ποιες στήλες χρησιμοποιούμε
    sigma_pred = parse_column(df["empirical_coverage"])
    error_true = parse_column(df["abs_coverage_diff"])

    return sigma_pred, error_true


def main():
    # 1. Path προς CSV με στατιστικά αβεβαιότητας
    # άλλαξε αν χρησιμοποιείς άλλο αρχείο
    csv_path = Path("drift_uncertainty_gaussian_coverage_stats.csv")

    # 2. Paths για να σώσουμε calibrators
    affine_model_path = Path("neural_uncertainty/affine_calibrator.pkl")
    hist_model_path = Path("neural_uncertainty/histogram_calibrator.pkl")

    sigma_pred, error_true = load_calibration_data(csv_path)

    # 3. Metrics BEFORE calibration
    metrics_unc = compute_calibration_metrics(sigma_pred, error_true)
    print("=== Uncalibrated uncertainty ===")
    print(f"ECE : {metrics_unc.ece:.4f}")
    print(f"MAE : {metrics_unc.mae:.4f}")
    print(f"MSE : {metrics_unc.mse:.4f}")
    print(f"NLL : {metrics_unc.nll:.4f}")

    # 4. Affine calibrator
    affine = AffineCalibrator()
    affine.fit(sigma_pred, error_true)
    sigma_affine = affine.predict(sigma_pred)

    metrics_aff = compute_calibration_metrics(sigma_affine, error_true)
    print("\n=== Affine calibrated ===")
    print(f"ECE : {metrics_aff.ece:.4f}")
    print(f"MAE : {metrics_aff.mae:.4f}")
    print(f"MSE : {metrics_aff.mse:.4f}")
    print(f"NLL : {metrics_aff.nll:.4f}")

    affine.save(str(affine_model_path))
    print(f"Saved affine calibrator to {affine_model_path}")

    # 5. Histogram calibrator
    hist = HistogramCalibrator(num_bins=15)
    hist.fit(sigma_pred, error_true)
    sigma_hist = hist.predict(sigma_pred)

    metrics_hist = compute_calibration_metrics(sigma_hist, error_true)
    print("\n=== Histogram calibrated ===")
    print(f"ECE : {metrics_hist.ece:.4f}")
    print(f"MAE : {metrics_hist.mae:.4f}")
    print(f"MSE : {metrics_hist.mse:.4f}")
    print(f"NLL : {metrics_hist.nll:.4f}")

    hist.save(str(hist_model_path))
    print(f"Saved histogram calibrator to {hist_model_path}")


if __name__ == "__main__":
    main()
