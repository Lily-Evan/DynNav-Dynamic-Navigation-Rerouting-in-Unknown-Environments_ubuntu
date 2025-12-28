from pathlib import Path

import numpy as np
import pandas as pd

from neural_uncertainty.calibration_methods import AffineCalibrator


def main():
    # 1. Paths
    learned_grid_path = Path("learned_uncertainty_grid.csv")
    calib_grid_path = Path("calib_learned_uncertainty_grid_affine.csv")
    affine_model_path = Path("neural_uncertainty/affine_calibrator_drift.pkl")
    print(f"Loading learned uncertainty grid from: {learned_grid_path}")

    # 2. Φόρτωση με pandas (γιατί το CSV έχει header / μη-αριθμητικές στήλες)
    df = pd.read_csv(learned_grid_path)
    print("Columns in learned_uncertainty_grid.csv:", df.columns.tolist())

    # Κρατάμε μόνο τις αριθμητικές στήλες (εκεί υποθέτουμε ότι βρίσκονται οι τιμές αβεβαιότητας)
    # Θέλουμε να κάνουμε calibrate ΜΟΝΟ τη στήλη learned_sigma
    if "learned_sigma" not in df.columns:
        raise RuntimeError("Column 'learned_sigma' not found in learned_uncertainty_grid.csv")

    numeric_cols = ["learned_sigma"]
    print("Numeric column to calibrate:", numeric_cols)

    values = df[numeric_cols].to_numpy()

    # 3. Φόρτωση εκπαιδευμένου calibrator
    calib = AffineCalibrator()
    calib.load(str(affine_model_path))
    print(f"Loaded affine calibrator from: {affine_model_path}")
    print(f"a = {calib.a:.4f}, b = {calib.b:.4f}")

    # 4. Εφαρμογή calibration (flatten → calibrate → reshape)
    flat = values.reshape(-1)
    flat_calib = calib.predict(flat)
    calib_values = flat_calib.reshape(values.shape)

    # 5. Επιστροφή των calibrated τιμών πίσω στο DataFrame
    df_calib = df.copy()
    df_calib[numeric_cols] = calib_values

    # 6. Αποθήκευση calibrated grid με ίδια δομή CSV
    df_calib.to_csv(calib_grid_path, index=False)
    print(f"Saved calibrated uncertainty grid at: {calib_grid_path}")
    print(f"Calibrated numeric grid shape: {calib_values.shape}")


if __name__ == "__main__":
    main()
