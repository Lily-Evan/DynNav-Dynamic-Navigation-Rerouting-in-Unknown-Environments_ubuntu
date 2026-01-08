import os
import pandas as pd
from nbv_option_entropy import option_entropy_score

CSV_IN = "nbv_frontier_scores_log.csv"   # το νέο μεγάλο log
CSV_OUT = "nbv_frontier_scores_log_with_entropy.csv"

TOP_K = 10
TEMP = 1.0

def main():
    if not os.path.exists(CSV_IN):
        raise FileNotFoundError(f"Missing {CSV_IN}. (Πρώτα κάνε exporter να γράφει log με seed/episode/step)")

    df = pd.read_csv(CSV_IN)

    for c in ["seed", "episode", "step", "R_local"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {CSV_IN}. cols={df.columns.tolist()}")

    ent = {}
    for key, g in df.groupby(["seed", "episode", "step"]):
        top = g.sort_values("R_local", ascending=False).head(TOP_K)
        H = option_entropy_score(top["R_local"].to_numpy(), temp=TEMP, normalize=True)
        ent[key] = H

    df["option_entropy_norm"] = df.apply(
        lambda r: ent[(r["seed"], r["episode"], r["step"])], axis=1
    )
    df["option_entropy_K"] = TOP_K
    df["option_entropy_temp"] = TEMP
    df["option_entropy_utility"] = "R_local"

    # optional composite score per row (για re-ranking)
    if "score_IG_I_R" in df.columns:
        W_OPT = 0.5
        df["score_IG_I_R_opt"] = df["score_IG_I_R"] + W_OPT * df["option_entropy_norm"]

    df.to_csv(CSV_OUT, index=False)
    print("[OK] wrote", CSV_OUT)

if __name__ == "__main__":
    main()
