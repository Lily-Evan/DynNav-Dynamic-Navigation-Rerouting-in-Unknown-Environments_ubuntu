# ================================================================
# Ablation Study: Multiple Seeds for Online Adaptation
# ================================================================

import os
import numpy as np
import pandas as pd
import torch

from online_drift_uncertainty_exp import OnlineDriftUncertaintyAdapterExp

FEATURE_COLS = ["entropy", "local_uncertainty", "speed"]
TARGET_COL = "drift"


def load_drift_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return X, y


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def run_single_seed(X, y, seed: int, device: str, dropout_p: float = 0.3):
    np.random.seed(seed)
    torch.manual_seed(seed)

    N = len(X)
    N0 = int(0.4 * N)
    X_base, y_base = X[:N0], y[:N0]
    X_online, y_online = X[N0:], y[N0:]

    adapter = OnlineDriftUncertaintyAdapterExp(
        model_path=None,
        input_dim=X.shape[1],
        lr=1e-4,
        device=device,
        max_buffer_size=512,
        weight_decay=1e-6,
        dropout_p=dropout_p,
    )

    y_pred_base = adapter.predict(X_base).reshape(-1)
    base_mse = mse(y_pred_base, y_base)

    mse_before = []
    mse_after = []

    for i in range(X_online.shape[0]):
        x_i = X_online[i]
        y_i = y_online[i]

        pred_b = adapter.predict(x_i).reshape(-1)[0]
        adapter.add_observation(x_i, np.array(y_i, dtype=np.float32))
        if (i + 1) % 8 == 0:
            _ = adapter.online_update(batch_size=32)
        pred_a = adapter.predict(x_i).reshape(-1)[0]

        mse_before.append((pred_b - y_i) ** 2)
        mse_after.append((pred_a - y_i) ** 2)

    return {
        "seed": seed,
        "base_mse": base_mse,
        "online_mse_before": float(np.mean(mse_before)),
        "online_mse_after": float(np.mean(mse_after)),
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "drift_dataset.csv")

    X, y = load_drift_dataset(csv_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = [0, 1, 2, 3, 4]
    results = []

    for s in seeds:
        print(f"[ABL-INFO] Running seed={s}")
        metrics = run_single_seed(X, y, seed=s, device=device, dropout_p=0.3)
        results.append(metrics)

    df = pd.DataFrame(results)
    print("\n[ABL-RESULTS] Per-seed metrics:")
    print(df)

    print("\n[ABL-SUMMARY]")
    for col in ["base_mse", "online_mse_before", "online_mse_after"]:
        print(f"{col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}")

    # ---- Save per-seed results and summary to CSV ----
    out_dir = os.path.join(base_dir, "logs_ablation")
    os.makedirs(out_dir, exist_ok=True)

    per_seed_path = os.path.join(out_dir, "online_adaptation_ablation_per_seed.csv")
    df.to_csv(per_seed_path, index=False)

    summary_rows = []
    for col in ["base_mse", "online_mse_before", "online_mse_after"]:
        summary_rows.append({
            "metric": col,
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "online_adaptation_ablation_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print(f"\n[ABL-INFO] Saved per-seed CSV to: {per_seed_path}")
    print(f"[ABL-INFO] Saved summary CSV to: {summary_path}")


if __name__ == "__main__":
    main()
