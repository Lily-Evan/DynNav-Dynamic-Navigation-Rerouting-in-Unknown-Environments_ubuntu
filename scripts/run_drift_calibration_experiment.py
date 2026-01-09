from pathlib import Path

import numpy as np
import pandas as pd

from neural_uncertainty.calibration_methods import (
    AffineCalibrator,
    HistogramCalibrator,
    compute_calibration_metrics,
)



DRIFT_DATASET_PATH = Path("drift_dataset.csv")

# ΤΩΡΑ με βάση τις πραγματικές στήλες:
PRED_COL = "local_uncertainty"   # predicted drift uncertainty
ERROR_COL = "drift"              # true drift error


def inspect_columns():
    df = pd.read_csv(DRIFT_DATASET_PATH)
    print("Columns in drift_dataset.csv:")
    print(df.columns.tolist())


def load_calibration_data():
    df = pd.read_csv(DRIFT_DATASET_PATH)

    sigma_pred = df[PRED_COL].to_numpy(dtype=float)
    error_true = df[ERROR_COL].to_numpy(dtype=float)

    return sigma_pred, error_true


def main():
    # Βήμα 0: Εκτύπωσε στήλες για να βεβαιωθείς τι υπάρχει
    print(">>> Inspecting drift_dataset columns...")
    inspect_columns()
    print(f"\nUsing PRED_COL = '{PRED_COL}', ERROR_COL = '{ERROR_COL}'")
    print(f"Loading data from: {DRIFT_DATASET_PATH}\n")

    sigma_pred, error_true = load_calibration_data()

    # 1. Metrics πριν την βαθμονόμηση
    metrics_unc = compute_calibration_metrics(sigma_pred, error_true)
    print("=== Uncalibrated drift uncertainty ===")
    print(f"ECE : {metrics_unc.ece:.4f}")
    print(f"MAE : {metrics_unc.mae:.4f}")
    print(f"MSE : {metrics_unc.mse:.4f}")
    print(f"NLL : {metrics_unc.nll:.4f}")

    # 2. Affine Calibrator
    affine = AffineCalibrator()
    affine.fit(sigma_pred, error_true)
    sigma_aff = affine.predict(sigma_pred)

    metrics_aff = compute_calibration_metrics(sigma_aff, error_true)
    print("\n=== Affine calibrated (drift) ===")
    print(f"ECE : {metrics_aff.ece:.4f}")
    print(f"MAE : {metrics_aff.mae:.4f}")
    print(f"MSE : {metrics_aff.mse:.4f}")
    print(f"NLL : {metrics_aff.nll:.4f}")

    affine_path = Path("neural_uncertainty/affine_calibrator_drift.pkl")
    affine.save(str(affine_path))
    print(f"Saved affine drift calibrator to {affine_path}")

    # 3. Histogram Calibrator
    hist = HistogramCalibrator(num_bins=15)
    hist.fit(sigma_pred, error_true)
    sigma_hist = hist.predict(sigma_pred)

    metrics_hist = compute_calibration_metrics(sigma_hist, error_true)
    print("\n=== Histogram calibrated (drift) ===")
    print(f"ECE : {metrics_hist.ece:.4f}")
    print(f"MAE : {metrics_hist.mae:.4f}")
    print(f"MSE : {metrics_hist.mse:.4f}")
    print(f"NLL : {metrics_hist.nll:.4f}")

    hist_path = Path("neural_uncertainty/histogram_calibrator_drift.pkl")
    hist.save(str(hist_path))
    print(f"Saved histogram drift calibrator to {hist_path}")


if __name__ == "__main__":
    main()
