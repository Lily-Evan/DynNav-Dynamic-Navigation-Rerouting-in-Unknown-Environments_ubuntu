import os
import pandas as pd
import matplotlib.pyplot as plt

#CSV_IN = "ids_cost_shaping_results.csv"
CSV_IN = "ids_cost_shaping_results_real.csv"
P1 = "ids_cost_shaping_tradeoff.png"
P2 = "ids_cost_shaping_success.png"

def main():
    if not os.path.exists(CSV_IN):
        raise FileNotFoundError(f"Missing {CSV_IN}. Run run_ids_cost_shaping_demo.py first.")

    df = pd.read_csv(CSV_IN)

    # Aggregate
    g = df.groupby(["attack_level", "mode"]).mean(numeric_only=True).reset_index()

    # Plot tradeoff: total_cost vs total_risk (one line per mode)
    plt.figure()
    for mode in sorted(g["mode"].unique()):
        sub = g[g["mode"] == mode].sort_values("attack_level")
        plt.plot(sub["total_cost"], sub["total_risk"], marker="o", label=mode)
    plt.xlabel("mean total_cost")
    plt.ylabel("mean total_risk")
    plt.title("IDS cost-shaping tradeoff (cost vs risk)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(P1, dpi=200)
    print(f"[OK] Saved {P1}")

    # Plot success vs attack level
    plt.figure()
    for mode in sorted(g["mode"].unique()):
        sub = g[g["mode"] == mode].sort_values("attack_level")
        plt.plot(sub["attack_level"], sub["episode_success"], marker="o", label=mode)
    plt.xlabel("attack_level")
    plt.ylabel("mean episode_success")
    plt.title("Success under attack: baseline vs shaping")
    plt.legend()
    plt.tight_layout()
    plt.savefig(P2, dpi=200)
    print(f"[OK] Saved {P2}")

if __name__ == "__main__":
    main()

