import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_CANDIDATES = [
    "returnability_mu_sweep.csv",
    "nbv_frontier_scores.csv",
    "nbv_goal_scores.csv",
    "nbv_frontier_top10.csv",
    "nbv_top10_full.csv",
]

OUT_CSV = "option_entropy_summary.csv"
PLOT_MU = "option_entropy_vs_mu.png"
PLOT_HIST = "option_entropy_hist.png"

def find_existing_csv():
    for f in CSV_CANDIDATES:
        if os.path.exists(f):
            return f
    return None

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def softmax(x, temp=1.0):
    x = np.asarray(x, dtype=float) / max(1e-9, float(temp))
    x = x - np.max(x)
    e = np.exp(x)
    s = e / (np.sum(e) + 1e-12)
    return s

def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))

def main():
    csv_in = find_existing_csv()
    if csv_in is None:
        raise FileNotFoundError(f"No input CSV found. Tried: {CSV_CANDIDATES}")

    df = pd.read_csv(csv_in)

    mu_col = pick_col(df, ["mu", "μ", "returnability_mu"])
    seed_col = pick_col(df, ["seed", "run", "trial"])
    episode_col = pick_col(df, ["episode", "ep", "world_id", "map_id"])

    ret_col = pick_col(df, ["returnability", "R", "meanR", "score_R", "return_score"])
    cost_col = pick_col(df, ["return_cost", "cost_return", "returnability_cost", "R_cost"])

    if ret_col is None and cost_col is None:
        for c in df.columns:
            if "return" in c.lower():
                ret_col = c
                break

    if ret_col is None and cost_col is None:
        raise ValueError(f"Could not find returnability/return-cost column in {csv_in}. Columns={df.columns.tolist()}")

    rank_col = pick_col(df, ["rank", "k", "topk_rank"])

    group_keys = []
    if mu_col: group_keys.append(mu_col)
    if seed_col: group_keys.append(seed_col)
    if episode_col: group_keys.append(episode_col)

    if not group_keys:
        df["_group"] = 0
        group_keys = ["_group"]

    K = 10
    TEMP = 1.0

    rows = []
    for gval, gdf in df.groupby(group_keys):
        sub = gdf.copy()

        if rank_col is not None:
            sub = sub.sort_values(rank_col).head(K)
        else:
            if ret_col is not None:
                sub = sub.sort_values(ret_col, ascending=False).head(K)
            else:
                sub = sub.sort_values(cost_col, ascending=True).head(K)

        if ret_col is not None:
            u = sub[ret_col].to_numpy(dtype=float)
        else:
            u = -sub[cost_col].to_numpy(dtype=float)

        p = softmax(u, temp=TEMP)
        H = entropy(p)
        Hn = H / np.log(len(p) + 1e-12)

        row = {
            "K": int(len(p)),
            "entropy": float(H),
            "entropy_norm": float(Hn),
            "csv_in": csv_in,
        }

        if isinstance(gval, tuple):
            for k, v in zip(group_keys, gval):
                row[k] = v
        else:
            row[group_keys[0]] = gval

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV} from {csv_in} rows={len(out)}")

    if mu_col and mu_col in out.columns:
        gg = out.groupby(mu_col)["entropy_norm"].mean().reset_index().sort_values(mu_col)
        plt.figure()
        plt.plot(gg[mu_col], gg["entropy_norm"], marker="o")
        plt.xlabel(mu_col)
        plt.ylabel("mean normalized option entropy")
        plt.title("Optionality vs returnability weight (mu)")
        plt.tight_layout()
        plt.savefig(PLOT_MU, dpi=200)
        print(f"[OK] saved {PLOT_MU}")
    else:
        print("[INFO] mu column not found; skipping option_entropy_vs_mu.png")

    plt.figure()
    plt.hist(out["entropy_norm"].to_numpy(dtype=float), bins=20)
    plt.xlabel("normalized option entropy")
    plt.ylabel("count")
    plt.title("Distribution of optionality (normalized entropy)")
    plt.tight_layout()
    plt.savefig(PLOT_HIST, dpi=200)
    print(f"[OK] saved {PLOT_HIST}")

if __name__ == "__main__":
    main()
