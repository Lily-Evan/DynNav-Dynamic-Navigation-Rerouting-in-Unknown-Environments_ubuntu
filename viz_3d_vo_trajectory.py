import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


if __name__ == "__main__":
    vo_csv = "vo_trajectory.csv"
    print(f"[VIZ3D] Loading VO trajectory from: {vo_csv}")
    df_vo = pd.read_csv(vo_csv)

    required = {"frame_idx", "x", "y", "z"}
    if not required.issubset(df_vo.columns):
        raise RuntimeError(f"{vo_csv} πρέπει να έχει στήλες {required}")

    xs = df_vo["x"].to_numpy()
    ys = df_vo["y"].to_numpy()
    zs = df_vo["z"].to_numpy()
    frames = df_vo["frame_idx"].to_numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    p = ax.scatter(xs, ys, zs, c=frames, cmap="viridis", s=5)
    fig.colorbar(p, ax=ax, label="Frame index")

    # highlight start / end
    ax.scatter(xs[0], ys[0], zs[0], color="red", s=50, label="Start")
    ax.scatter(xs[-1], ys[-1], zs[-1], color="orange", s=50, label="End")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D VO Trajectory (colored by time)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("vo_trajectory_3d.png", dpi=200)
    plt.show()
    print("[VIZ3D] Saved vo_trajectory_3d.png")
