# ================================================================
# Compare Uncertainty Calibration:
#   - Dropout-based model
#   - Ensemble-based model
#
# Διαβάζει τα δύο summary CSVs και τυπώνει/σώζει έναν πίνακα
# με MSE, NLL, UCE, num_samples.
# ================================================================

import os
import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dropout_dir = os.path.join(base_dir, "logs_calibration")
    ensemble_dir = os.path.join(base_dir, "logs_calibration_ensemble")

    dropout_csv = os.path.join(dropout_dir, "uncertainty_calibration_summary.csv")
    ensemble_csv = os.path.join(ensemble_dir, "uncertainty_calibration_summary_ensemble.csv")

    if not os.path.exists(dropout_csv):
        raise FileNotFoundError(f"Dropout summary not found: {dropout_csv}")
    if not os.path.exists(ensemble_csv):
        raise FileNotFoundError(f"Ensemble summary not found: {ensemble_csv}")

    df_drop = pd.read_csv(dropout_csv)
    df_ens = pd.read_csv(ensemble_csv)

    # Περιμένουμε 1 γραμμή ανά CSV
    row_drop = df_drop.iloc[0]
    row_ens = df_ens.iloc[0]

    data = [
        {
            "model": "dropout",
            "mse": row_drop["mse"],
            "nll": row_drop["nll"],
            "uce": row_drop["uce"],
            "num_samples": int(row_drop["num_samples"]),
        },
        {
            "model": "ensemble",
            "mse": row_ens["mse"],
            "nll": row_ens["nll"],
            "uce": row_ens["uce"],
            "num_samples": int(row_ens["num_samples"]),
        },
    ]

    df_comp = pd.DataFrame(data)

    print("\n============ UNCERTAINTY CALIBRATION COMPARISON ============")
    print(df_comp.to_string(index=False))
    print("=============================================================")

    # Optional: relative improvements (ensemble vs dropout)
    drop_mse = row_drop["mse"]
    ens_mse = row_ens["mse"]
    drop_nll = row_drop["nll"]
    ens_nll = row_ens["nll"]
    drop_uce = row_drop["uce"]
    ens_uce = row_ens["uce"]

    print("\n[RELATIVE COMPARISON] (ensemble relative to dropout)")
    print(f"MSE:  ensemble / dropout = {ens_mse / drop_mse:.3f}")
    print(f"NLL:  ensemble / dropout = {ens_nll / drop_nll:.3f}")
    print(f"UCE:  ensemble / dropout = {ens_uce / drop_uce:.3f}")

    # Save combined CSV
    out_dir = os.path.join(base_dir, "logs_calibration")
    out_path = os.path.join(out_dir, "uncertainty_calibration_comparison_dropout_vs_ensemble.csv")
    df_comp.to_csv(out_path, index=False)
    print(f"\n[COMPARE] Saved combined comparison CSV to: {out_path}")


if __name__ == "__main__":
    main()
