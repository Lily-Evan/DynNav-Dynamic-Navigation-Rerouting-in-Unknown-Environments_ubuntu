# ================================================================
# Real World Evaluation (using experimental online adapter)
# Baseline + Online Adaptation + Self-Trust
# ================================================================

import os
import numpy as np
import torch

from real_world_dataset_loader import RealWorldDatasetLoader
from online_drift_uncertainty_exp import OnlineDriftUncertaintyAdapterExp


FEATURE_COLS = ["entropy", "local_uncertainty", "speed"]
TARGET_COL = "drift"


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def compute_self_trust(error_sq: float, var: float, base_mse: float, base_var: float) -> float:
    """
    Απλός normalized self-trust δείκτης με βάση error & variance.
    """
    e_norm = error_sq / (base_mse + 1e-8)
    v_norm = var / (base_var + 1e-8)
    penalty = e_norm + v_norm
    S = 1.0 / (1.0 + penalty)
    return float(S)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "drift_dataset.csv")

    print(f"[RW-INFO] Loading real-world-style dataset from: {csv_path}")
    loader = RealWorldDatasetLoader(csv_path)

    X, y = loader.build_feature_matrix(FEATURE_COLS, TARGET_COL)

    N = len(X)
    N0 = int(0.4 * N)

    X_base = X[:N0]
    y_base = y[:N0]
    X_online = X[N0:]
    y_online = y[N0:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RW-INFO] Using device: {device}")

    # Χρησιμοποιούμε τον experimental adapter (ίδιος με τα άλλα exp scripts)
    adapter = OnlineDriftUncertaintyAdapterExp(
        model_path=None,              # ξεκινάμε από scratch
        input_dim=X.shape[1],
        lr=1e-4,
        device=device,
        max_buffer_size=512,
        weight_decay=1e-6,
        dropout_p=0.1,
    )

    # ============== Baseline ===================
    print("\n[RW-INFO] Baseline evaluation...")
    y_pred_base = adapter.predict(X_base).reshape(-1)
    base_mse = mse(y_pred_base, y_base)

    # baseline epistemic variance estimation
    var_base_list = []
    for i in range(X_base.shape[0]):
        _, var_b = adapter.predict_with_uncertainty(X_base[i], n_samples=10)
        var_base_list.append(var_b.reshape(-1)[0])
    base_var = float(np.mean(var_base_list))

    print(f"[RW-RESULT] Baseline MSE = {base_mse:.6f}")
    print(f"[RW-RESULT] Baseline VAR = {base_var:.6f}")

    # ============== Online Adaptation ===================
    print("\n[RW-INFO] Online adaptation evaluation...")

    mse_before = []
    mse_after = []
    S_values = []

    for i in range(X_online.shape[0]):
        x_i = X_online[i]
        y_i = y_online[i]

        # prediction BEFORE update
        mean_b, var_b = adapter.predict_with_uncertainty(x_i, n_samples=20)
        pred_b = mean_b.reshape(-1)[0]
        var_b_scalar = var_b.reshape(-1)[0]
        err_b_sq = (pred_b - y_i) ** 2

        adapter.add_observation(x_i, np.array(y_i, dtype=np.float32))
        if (i + 1) % 8 == 0:
            _ = adapter.online_update(batch_size=32)

        # prediction AFTER update
        mean_a, var_a = adapter.predict_with_uncertainty(x_i, n_samples=20)
        pred_a = mean_a.reshape(-1)[0]
        var_a_scalar = var_a.reshape(-1)[0]
        err_a_sq = (pred_a - y_i) ** 2

        mse_before.append(err_b_sq)
        mse_after.append(err_a_sq)

        # Self-trust based on AFTER prediction
        S = compute_self_trust(err_a_sq, var_a_scalar, base_mse, base_var)
        S_values.append(S)

        if (i + 1) % 50 == 0:
            print(
                f"[RW-STEP {i+1}/{X_online.shape[0]}] "
                f"MSEbefore={np.mean(mse_before[-50:]):.4f}  "
                f"MSEafter={np.mean(mse_after[-50:]):.4f}  "
                f"S={np.mean(S_values[-50:]):.3f}"
            )

    print("\n================ REAL-WORLD SUMMARY ================")
    print(f"Overall MSE BEFORE online adaptation : {np.mean(mse_before):.6f}")
    print(f"Overall MSE AFTER  online adaptation : {np.mean(mse_after):.6f}")
    print(f"Average Self-Trust S                 : {np.mean(S_values):.3f}")
    print("====================================================")


if __name__ == "__main__":
    main()

