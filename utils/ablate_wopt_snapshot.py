import numpy as np
import pandas as pd

CSV_IN = "nbv_scores_with_option_entropy.csv"
CSV_OUT = "wopt_ablation_snapshot.csv"

def main():
    df = pd.read_csv(CSV_IN)

    if "score_IG_I_R" not in df.columns or "option_entropy_norm" not in df.columns:
        raise ValueError("Need score_IG_I_R and option_entropy_norm")

    w_list = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    rows = []

    for w in w_list:
        score = df["score_IG_I_R"] + w * df["option_entropy_norm"]
        idx = int(np.argmax(score.to_numpy()))
        pick = df.iloc[idx][["y","x","IG","I_local","R_local","score_IG_I_R","option_entropy_norm"]].to_dict()
        pick["w_opt"] = w
        pick["picked_score"] = float(score.iloc[idx])
        rows.append(pick)

    out = pd.DataFrame(rows)
    out.to_csv(CSV_OUT, index=False)
    print("[OK] wrote", CSV_OUT)
    print(out[["w_opt","y","x","picked_score"]].to_string(index=False))

if __name__ == "__main__":
    main()
