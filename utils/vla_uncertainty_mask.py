import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiobj_planner import load_vla_config


def load_uncertainty_grid_df(csv_path: str):
    """
    Φορτώνει τον πίνακα coverage με uncertainty και γυρίζει:
      - df (raw dataframe)
      - U (2D grid με uncertainty)
    Περιμένουμε στήλες: row, col, uncertainty
    """
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

    return df, U


def build_region_mask(U: np.ndarray, region: str) -> np.ndarray:
    """
    Φτιάχνει 2D mask (True/False) στο grid με βάση το region από VLA:
      - full
      - top-left
      - top-right
      - bottom
      - high-uncertainty
    ΕΠΙΣΤΡΕΦΕΙ ΠΑΝΤΑ μάσκα ίδιου σχήματος με U: (rows, cols)
    """
    rows, cols = U.shape
    mask = np.zeros((rows, cols), dtype=bool)

    if region == "full":
        mask[:, :] = True
        return mask

    mid_row = rows // 2
    mid_col = cols // 2

    if region == "top-left":
        # πάνω (γραμμές από mid_row..τέλος), αριστερά (στήλες 0..mid_col)
        mask[mid_row:, : mid_col + 1] = True
        print("[VLA-U] Region = top-left")
    elif region == "top-right":
        mask[mid_row:, mid_col + 1 :] = True
        print("[VLA-U] Region = top-right")
    elif region == "bottom":
        # κάτω μέρος: γραμμές 0..mid_row
        mask[: mid_row + 1, :] = True
        print("[VLA-U] Region = bottom")
    elif region == "high-uncertainty":
        # κρατάμε κελιά με uncertainty πάνω από median
        median_u = np.nanmedian(U)
        mask = U >= median_u
        print(f"[VLA-U] Region = high-uncertainty, median U = {median_u:.4f}")
    else:
        print(f"[VLA-U] Unknown region '{region}', fallback σε full.")
        mask[:, :] = True

    return mask


def get_scaling_from_priority(priority: str):
    """
    Priority → (inside_factor, outside_factor)
    - entropy      → μειώνουμε U στη region, αφήνουμε σχεδόν ίδιο εκτός
    - uncertainty  → αυξάνουμε U στη region, μειώνουμε λίγο εκτός
    - combined     → ήπια ενίσχυση στη region, ήπια μείωση εκτός
    """
    if priority == "entropy":
        inside = 0.8
        outside = 1.0
    elif priority == "uncertainty":
        inside = 1.3
        outside = 0.8
    else:  # combined ή οτιδήποτε άλλο
        inside = 1.1
        outside = 0.9
    return inside, outside


if __name__ == "__main__":
    coverage_unc_csv = "coverage_grid_with_uncertainty.csv"
    output_csv = "coverage_grid_with_uncertainty_vla.csv"
    output_fig = "uncertainty_vla_mask.png"

    print(f"[VLA-U] Loading coverage with uncertainty from: {coverage_unc_csv}")
    df_cov, U = load_uncertainty_grid_df(coverage_unc_csv)

    rows, cols = U.shape

    # === φορτώνουμε VLA config ===
    cfg = load_vla_config()
    region = cfg.get("region", "full")
    priority = cfg.get("priority", "combined")
    print(f"[VLA-U] Using VLA config: region={region}, priority={priority}")

    # === χτίζουμε region mask ===
    region_mask = build_region_mask(U, region)

    # === επιλέγουμε scaling factors από priority ===
    inside_factor, outside_factor = get_scaling_from_priority(priority)
    print(f"[VLA-U] Scaling factors: inside={inside_factor}, outside={outside_factor}")

    # === φτιάχνουμε U_mod ===
    U_mod = U.copy()
    inside_idx = region_mask
    outside_idx = ~region_mask

    U_mod[inside_idx] = U_mod[inside_idx] * inside_factor
    U_mod[outside_idx] = U_mod[outside_idx] * outside_factor

    # clip στο [0,1]
    U_mod = np.clip(U_mod, 0.0, 1.0)

    # === ενημερώνουμε το df με τις νέες τιμές uncertainty ===
    new_unc = []
    for _, r in df_cov.iterrows():
        row = int(r["row"])
        col = int(r["col"])
        new_unc.append(U_mod[row, col])

    df_out = df_cov.copy()
    df_out["uncertainty"] = new_unc

    df_out.to_csv(output_csv, index=False)
    print(f"[VLA-U] Saved VLA-adjusted uncertainty map to: {output_csv}")

    # === Visualization πριν / μετά ===
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    im1 = plt.imshow(U, origin="lower", cmap="magma")
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.title("Original uncertainty")
    plt.xlabel("col")
    plt.ylabel("row")

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(U_mod, origin="lower", cmap="magma")
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.title(f"VLA-adjusted uncertainty\n(region={region}, priority={priority})")
    plt.xlabel("col")
    plt.ylabel("row")

    plt.tight_layout()
    plt.savefig(output_fig, dpi=220)
    plt.show()
    print(f"[VLA-U] Saved uncertainty visualization to: {output_fig}")

    print("[VLA-U] Done.")
