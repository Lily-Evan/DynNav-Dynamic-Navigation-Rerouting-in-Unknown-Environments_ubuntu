import os
import numpy as np
import pandas as pd

from nbv_option_entropy import option_entropy_score

# =========================
# Input / Output
# =========================
CSV_IN = "nbv_frontier_scores.csv"
CSV_OUT = "nbv_scores_with_option_entropy.csv"

# =========================
# Parameters
# =========================
TOP_K = 10
TEMP = 1.0

def main():
    if not os.path.exists(CSV_IN):
        raise FileNotFoundError(f"Missing {CSV_IN}")

    df = pd.read_csv(CSV_IN)

    # -------------------------
    # Required columns (KNOWN)
    # -------------------------
    required = ["IG", "R_local"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found. Columns={df.columns.tolist()}")

    # -------------------------
    # Compute option entropy
    # -------------------------
    # We assume this CSV corresponds to ONE NBV decision
    # (set of frontier candidates at one step)
    # Utility = returnability (R_local)
    # Higher R_local = safer / more returnable option
    # Entropy measures OPTIONALITY of decision

    df_sorted = df.sort_values("R_local", ascending=False).head(TOP_K)
    utilities = df_sorted["R_local"].to_numpy(dtype=float)

    H_opt = option_entropy_score(
        utilities,
        temp=TEMP,
        normalize=True
    )

    # -------------------------
    # Attach entropy to ALL rows
    # (decision-level metric)
    # -------------------------
    df["option_entropy_norm"] = H_opt
    df["option_entropy_K"] = TOP_K
    df["option_entropy_temp"] = TEMP
    df["option_entropy_utility"] = "R_local"

    # -------------------------
    # Optional: new composite score
    # Existing best score in your file:
    # score_IG_I_R
    # -------------------------
    if "score_IG_I_R" in df.columns:
        W_OPT = 0.5
        df["score_IG_I_R_opt"] = (
            df["score_IG_I_R"] + W_OPT * df["option_entropy_norm"]
        )
        print("[INFO] Added composite score: score_IG_I_R_opt")

    df.to_csv(CSV_OUT, index=False)

    print("[OK] wrote", CSV_OUT)
    print("[INFO] option_entropy_norm =", H_opt)
    print("[INFO] used TOP_K =", TOP_K)
    print("[INFO] utility column = R_local")

if __name__ == "__main__":
    main()
