# ================================================================
# Uncertainty Calibration Preparation
# - Κάνει ένα απλό online "training" στο ~70% των δειγμάτων
# - Υπολογίζει mean & variance predictions στο υπόλοιπο ~30%
# - Σώζει arrays για calibration αξιολόγηση
# ================================================================

import os
import numpy as np
import torch

from real_world_dataset_loader import RealWorldDatasetLoader
from online_drift_uncertainty_exp import OnlineDriftUncertaintyAdapterExp

FEATURE_COLS = ["entropy", "local_uncertainty", "speed"]
TARGET_COL = "drift"


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "drift_dataset.csv")

    print(f"[CAL-PREP] Loading dataset from: {csv_path}")
    loader = RealWorldDatasetLoader(csv_path)
    X, y = loader.build_feature_matrix(FEATURE_COLS, TARGET_COL)

    N = len(X)
    N_train = int(0.7 * N)
    X_train, y_train = X[:N_train], y[:N_train]
    X_test, y_test = X[N_train:], y[N_train:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CAL-PREP] Using device: {device}")

    adapter = OnlineDriftUncertaintyAdapterExp(
        model_path=None,
        input_dim=X.shape[1],
        lr=1e-4,
        device=device,
        max_buffer_size=512,
        weight_decay=1e-6,
        dropout_p=0.1,
    )

    # ----------------- "Training" phase (online style) -----------------
    print(f"[CAL-PREP] Online training on {N_train} samples...")

    for i in range(N_train):
        x_i = X_train[i]
        y_i = y_train[i]
        adapter.add_observation(x_i, np.array(y_i, dtype=np.float32))
        if (i + 1) % 16 == 0:
            _ = adapter.online_update(batch_size=64)

    print("[CAL-PREP] Training phase done.")

    # ----------------- Calibration test phase -----------------
    print(f"[CAL-PREP] Evaluating on calibration test set: {len(X_test)} samples...")

    means = []
    vars_ = []
    y_list = []

    for i in range(len(X_test)):
        x_i = X_test[i]
        y_i = y_test[i]

        mean_i, var_i = adapter.predict_with_uncertainty(x_i, n_samples=30)
        means.append(mean_i.reshape(-1)[0])
        vars_.append(var_i.reshape(-1)[0])
        y_list.append(float(y_i))

    means = np.array(means, dtype=np.float32)
    vars_ = np.array(vars_, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)

    out_dir = os.path.join(base_dir, "logs_calibration")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "means.npy"), means)
    np.save(os.path.join(out_dir, "vars.npy"), vars_)
    np.save(os.path.join(out_dir, "y_true.npy"), y_arr)

    print(f"[CAL-PREP] Saved means/vars/y_true to: {out_dir}")
    print(f"[CAL-PREP] means.shape={means.shape}, vars.shape={vars_.shape}, y.shape={y_arr.shape}")


if __name__ == "__main__":
    main()
