import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from multiobj_planner import load_vla_config


def load_replan_path(csv_path: str):
    """
    Φορτώνει replan waypoints.
    Περιμένουμε στήλες: id, x, z
    """
    df = pd.read_csv(csv_path)
    required = {"x", "z"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"{csv_path} πρέπει να έχει στήλες {required}")
    xs = df["x"].to_numpy()
    zs = df["z"].to_numpy()
    return df, xs, zs


def moving_average_smooth(x: np.ndarray, window: int) -> np.ndarray:
    """
    Απλό 1D smoothing με moving average.
    """
    if window < 2:
        return x.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def pick_window_from_priority(priority: str) -> int:
    """
    Ανάλογα με το priority από VLA:
      - 'entropy'      → πιο δυνατός εξομαλυντικός (μεγάλο window)
      - 'uncertainty'  → ελαφρύ smoothing
      - 'combined'     → ενδιάμεσο
    """
    if priority == "entropy":
        return 21
    elif priority == "uncertainty":
        return 7
    else:  # combined ή οτιδήποτε άλλο
        return 13


if __name__ == "__main__":
    replan_csv = "replan_waypoints.csv"
    out_csv = "replan_waypoints_smooth_vla.csv"
    out_fig = "vla_path_smoothing.png"

    print(f"[VLA-SMOOTH] Loading replan path from: {replan_csv}")
    df_wp, xs, zs = load_replan_path(replan_csv)

    # === Load VLA config ===
    cfg = load_vla_config()
    priority = cfg.get("priority", "combined")
    region = cfg.get("region", "full")
    print(f"[VLA-SMOOTH] Using VLA config: priority={priority}, region={region}")

    # επιλέγουμε window size με βάση το priority
    window = pick_window_from_priority(priority)
    print(f"[VLA-SMOOTH] Smoothing window size = {window}")

    # === Apply smoothing on x,z separately ===
    xs_smooth = moving_average_smooth(xs, window=window)
    zs_smooth = moving_average_smooth(zs, window=window)

    # φτιάχνουμε νέο DataFrame για αποθήκευση
    df_smooth = df_wp.copy()
    df_smooth["x"] = xs_smooth
    df_smooth["z"] = zs_smooth

    df_smooth.to_csv(out_csv, index=False)
    print(f"[VLA-SMOOTH] Saved VLA-smoothed path to: {out_csv}")

    # === Visualization: original vs smoothed path ===
    plt.figure(figsize=(7, 6))
    plt.plot(xs, zs, "-", linewidth=0.8, alpha=0.5, label="Original replan path")
    plt.plot(xs_smooth, zs_smooth, "-", linewidth=1.5, label="VLA-smoothed path")

    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(f"VLA-based path smoothing (priority={priority})")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=220)
    plt.show()
    print(f"[VLA-SMOOTH] Saved plot to: {out_fig}")

