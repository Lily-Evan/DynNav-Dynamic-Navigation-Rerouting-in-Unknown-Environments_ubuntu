import pandas as pd

CSV_IN = "nbv_scores_with_option_entropy.csv"

def main():
    df = pd.read_csv(CSV_IN)

    if "score_IG_I_R" not in df.columns or "score_IG_I_R_opt" not in df.columns:
        raise ValueError("Missing score columns. Θέλω score_IG_I_R και score_IG_I_R_opt")

    # rank baseline
    df_b = df.sort_values("score_IG_I_R", ascending=False).reset_index(drop=True)
    df_b["rank_baseline"] = df_b.index + 1

    # rank opt
    df_o = df.sort_values("score_IG_I_R_opt", ascending=False).reset_index(drop=True)
    df_o["rank_opt"] = df_o.index + 1

    # merge back by (y,x) since these identify candidate
    m = df_b[["y","x","rank_baseline","score_IG_I_R"]].merge(
        df_o[["y","x","rank_opt","score_IG_I_R_opt"]],
        on=["y","x"],
        how="inner"
    )

    top_b = df_b.iloc[0][["y","x","score_IG_I_R"]].to_dict()
    top_o = df_o.iloc[0][["y","x","score_IG_I_R_opt"]].to_dict()

    print("=== TOP-1 baseline (score_IG_I_R) ===")
    print(top_b)
    print("\n=== TOP-1 optionality (score_IG_I_R_opt) ===")
    print(top_o)

    # how much ranks moved
    m["delta_rank"] = m["rank_baseline"] - m["rank_opt"]
    print("\n=== Rank movement summary ===")
    print(m["delta_rank"].describe().to_string())

    # show biggest movers
    print("\n=== Biggest movers (|delta_rank|) top 10 ===")
    mm = m.copy()
    mm["abs"] = mm["delta_rank"].abs()
    print(mm.sort_values("abs", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()
