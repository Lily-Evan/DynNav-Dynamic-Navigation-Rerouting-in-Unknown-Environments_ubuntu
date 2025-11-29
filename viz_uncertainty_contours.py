import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_uncertainty_grid(csv_path: str):
    df = pd.read_csv(csv_path)
    required = {"row", "col", "uncertainty"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"{csv_path} πρέπει να έχει στήλες {required}")

    max_row = int(df["row"].max())
    max_col = int(df["col"].max())
    U = np.zeros((max_row + 1, max_col + 1), dtype=float)
    U[:] = np.nan

    for _, r in df.iterrows():
        row = int(r["row"])
        col = int(r["col"])
        u = float(r["uncertainty"])
        U[row, col] = u

    return U


if __name__ == "__main__":
    cov_unc_csv = "coverage_grid_with_uncertainty.csv"
    vo_csv = "vo_trajectory.csv"  # προαιρετικά

    print(f"[VIZ-U] Loading uncertainty grid from: {cov_unc_csv}")
    U = load_uncertainty_grid(cov_unc_csv)
    rows, cols = U.shape

    # προαιρετικά: φορτώνουμε VO μόνο για overlay σε x-z plane
    try:
        df_vo = pd.read_csv(vo_csv)
        has_vo = {"x", "z"}.issubset(df_vo.columns)
    except FileNotFoundError:
        has_vo = False

    plt.figure(figsize=(7, 6))
    im = plt.imshow(U, origin="lower", cmap="magma")
    plt.colorbar(im, label="Uncertainty")

    # Contours
    levels = np.linspace(np.nanmin(U), np.nanmax(U), 6)
    plt.contour(U, levels=levels, colors="white", linewidths=0.7, origin="lower")

    if has_vo:
        xs = df_vo["x"].to_numpy()
        zs = df_vo["z"].to_numpy()

        # κλιμάκωση για να ταιριάξει σε grid coords
        x_min, x_max = xs.min(), xs.max()
        z_min, z_max = zs.min(), zs.max()
        if abs(x_max - x_min) < 1e-9:
            x_max = x_min + 1.0
        if abs(z_max - z_min) < 1e-9:
            z_max = z_min + 1.0

        cols_vo = (xs - x_min) / (x_max - x_min) * (cols - 1)
        rows_vo = (zs - z_min) / (z_max - z_min) * (rows - 1)

        plt.plot(cols_vo, rows_vo, "-", linewidth=1.0, color="cyan", label="VO (proj)")
        plt.legend(loc="upper right")

    plt.xlabel("col")
    plt.ylabel("row")
    plt.title("Uncertainty heatmap with contours")
    plt.tight_layout()
    plt.savefig("uncertainty_contours.png", dpi=200)
    plt.show()
    print("[VIZ-U] Saved uncertainty_contours.png")
