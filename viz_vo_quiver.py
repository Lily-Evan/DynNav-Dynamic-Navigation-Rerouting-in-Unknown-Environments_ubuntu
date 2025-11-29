import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    vo_csv = "vo_trajectory.csv"
    print(f"[VIZ-Q] Loading VO trajectory from: {vo_csv}")
    df_vo = pd.read_csv(vo_csv)

    required = {"x", "z"}
    if not required.issubset(df_vo.columns):
        raise RuntimeError(f"{vo_csv} πρέπει να έχει στήλες {required}")

    xs = df_vo["x"].to_numpy()
    zs = df_vo["z"].to_numpy()

    # Υπολογισμός διαφόρων (dx, dz)
    dx = np.diff(xs)
    dz = np.diff(zs)

    # για να μην είναι τεράστιο, κάνουμε decimate (π.χ. κάθε 5ο δείγμα)
    step = 5
    xs_q = xs[::step]
    zs_q = zs[::step]
    dx_q = dx[::step]
    dz_q = dz[::step]

    plt.figure(figsize=(7, 6))
    plt.plot(xs, zs, "-", linewidth=0.8, label="VO path")
    plt.quiver(xs_q, zs_q, dx_q, dz_q, angles="xy", scale_units="xy", scale=1.0,
               width=0.0025, alpha=0.8, label="Direction")

    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("VO trajectory with direction arrows")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("vo_quiver.png", dpi=200)
    plt.show()
    print("[VIZ-Q] Saved vo_quiver.png")
