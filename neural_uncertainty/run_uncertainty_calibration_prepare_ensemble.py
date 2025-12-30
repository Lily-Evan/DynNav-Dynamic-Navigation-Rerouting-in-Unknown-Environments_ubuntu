# ================================================================
# Uncertainty Calibration Preparation (Ensemble Version)
# ================================================================

import os
import numpy as np
import torch

from real_world_dataset_loader import RealWorldDatasetLoader
from ensemble_drift_uncertainty_exp import EnsembleDriftUncertaintyAdapterExp

FEATURE_COLS = ["entropy", "local_uncertainty", "speed"]
TARGET_COL = "drift"


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "drift_dataset.csv")

    print(f"[CAL-PREP-ENS] Loading dataset from: {csv_path}")
    loader = RealWorldDatasetLoader(csv_path)
    X, y = loader.build_feature_matrix(FEATURE_COLS, TARGET_COL)

    N = len(X)
    N_train = int(0.7 * N)
    X_train, y_train = X[:N_train], y[:N_train]
    X_test, y_test = X[N_train:], y[N_train:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CAL-PREP-ENS] Using device: {device}")

    ensemble = EnsembleDriftUncertaintyAdapterExp(
        n_members=5,
        input_dim=X.shape[1],
        lr=1e-4,
        device=device,
        max_buffer_size=512,
        weight_decay=1e-6,
        dropout_p=0.1,
        base_model_path=None,
        base_seed=0,
    )

    # ----------------- Training phase -----------------
    print(f"[CAL-PREP-ENS] Online training on {N_train} samples (ensemble)...")

    for i in range(N_train):
        x_i = X_train[i]
        y_i = y_train[i]
        ensemble.add_observation(x_i, np.array(y_i, dtype=np.float32))
        if (i + 1) % 16 == 0:
            _ = ensemble.online_update(batch_size=64)

    print("[CAL-PREP-ENS] Training phase done.")

    # ----------------- Calibration test phase -----------------
    print(f"[CAL-PREP-ENS] Evaluating on calibration test set: {len(X_test)} samples...")

    means = []
    vars_ = []
    y_list = []

    for i in range(len(X_test)):
        x_i = X_test[i]
        y_i = y_test[i]

        mean_i, var_i = ensemble.predict_with_uncertainty(x_i, n_mc=0)
        means.append(mean_i.reshape(-1)[0])
        vars_.append(var_i.reshape(-1)[0])
        y_list.append(float(y_i))

    means = np.array(means, dtype=np.float32)
    vars_ = np.array(vars_, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)

    out_dir = os.path.join(base_dir, "logs_calibration_ensemble")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "means.npy"), means)
    np.save(os.path.join(out_dir, "vars.npy"), vars_)
    np.save(os.path.join(out_dir, "y_true.npy"), y_arr)

    print(f"[CAL-PREP-ENS] Saved means/vars/y_true to: {out_dir}")
    print(f"[CAL-PREP-ENS] means.shape={means.shape}, vars.shape={vars_.shape}, y.shape={y_arr.shape}")


if __name__ == "__main__":
    main()
