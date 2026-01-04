# plot_irreversibility_tau_sweep.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    csv_path = "irreversibility_bottleneck_tau_sweep.csv"
    df = pd.read_csv(csv_path)

    # Basic info
    i_start = df["I_start"].iloc[0]
    i_goal = df["I_goal"].iloc[0]
    print(f"I_start={i_start:.3f}, I_goal={i_goal:.3f}")

    # Split success/fail
    df_s = df[df["success"] == 1].copy()
    df_f = df[df["success"] == 0].copy()

    # 1) Success vs tau (step-like)
    plt.figure()
    plt.plot(df["tau"], df["success"], marker="o")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("tau (irreversibility threshold)")
    plt.ylabel("success (0/1)")
    plt.title("Feasibility vs irreversibility threshold τ")
    plt.grid(True, alpha=0.3)
    out1 = "irreversibility_success_vs_tau.png"
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    print("Saved:", out1)
    plt.close()

    # 2) Expansions vs tau (only successes)
    plt.figure()
    if len(df_s) > 0:
        plt.plot(df_s["tau"], df_s["expansions"], marker="o")
    plt.xlabel("tau (irreversibility threshold)")
    plt.ylabel("A* expansions")
    plt.title("Search effort vs τ (successful runs)")
    plt.grid(True, alpha=0.3)
    out2 = "irreversibility_expansions_vs_tau.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    print("Saved:", out2)
    plt.close()

    # 3) Cost vs tau (only successes)
    plt.figure()
    if len(df_s) > 0:
        plt.plot(df_s["tau"], df_s["cost"], marker="o")
    plt.xlabel("tau (irreversibility threshold)")
    plt.ylabel("path cost")
    plt.title("Path cost vs τ (successful runs)")
    plt.grid(True, alpha=0.3)
    out3 = "irreversibility_cost_vs_tau.png"
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    print("Saved:", out3)
    plt.close()

    # 4) Max I on path vs tau (only successes)
    plt.figure()
    if len(df_s) > 0:
        plt.plot(df_s["tau"], df_s["max_I_on_path"], marker="o")
    plt.xlabel("tau (irreversibility threshold)")
    plt.ylabel("max I along path")
    plt.title("Constraint tightness (max I on path) vs τ")
    plt.grid(True, alpha=0.3)
    out4 = "irreversibility_maxI_vs_tau.png"
    plt.savefig(out4, dpi=200, bbox_inches="tight")
    print("Saved:", out4)
    plt.close()

    # Print a small scientific summary
    if len(df_s) > 0:
        tau_min_success = df_s["tau"].min()
        print(f"\nCritical feasibility threshold ≈ {tau_min_success:.2f}")
        print("Failure reasons (counts):")
        print(df_f["reason"].value_counts().to_string())
    else:
        print("\nNo successful runs in this sweep.")


if __name__ == "__main__":
    main()
