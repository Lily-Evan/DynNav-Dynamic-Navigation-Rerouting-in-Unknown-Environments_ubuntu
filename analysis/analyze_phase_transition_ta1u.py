import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_IN = "irreversibility_tau_sweep.csv"
CSV_OUT = "phase_transition_tau_summary.csv"
PLOT_DER = "phase_transition_tau_derivative.png"
PLOT_SUCC = "phase_transition_tau_success.png"

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Missing required column. Tried {candidates}. Found: {df.columns.tolist()}")

def main():
    if not os.path.exists(CSV_IN):
        raise FileNotFoundError(f"Missing {CSV_IN}")

    df = pd.read_csv(CSV_IN)

    tau_col = pick_col(df, ["tau", "τ", "threshold_tau", "tau0"])
    succ_col = None
    for cand in ["success_rate", "success", "feasible_rate", "feasibility", "feasible"]:
        if cand in df.columns:
            succ_col = cand
            break
    if succ_col is None:
        raise ValueError(f"Could not find success column in {df.columns.tolist()}")

    # mean across seeds if needed
    g = df.groupby(tau_col)[succ_col].mean().reset_index().sort_values(tau_col)
    tau = g[tau_col].to_numpy(dtype=float)
    succ = g[succ_col].to_numpy(dtype=float)

    # numerical derivative
    dsucc = np.gradient(succ, tau)
    absd = np.abs(dsucc)
    idx = int(np.argmax(absd))
    tau_star = float(tau[idx])

    # half-peak width on |derivative|
    half = 0.5 * absd[idx]
    L = idx
    while L > 0 and absd[L] >= half:
        L -= 1
    R = idx
    while R < len(absd) - 1 and absd[R] >= half:
        R += 1
    width = float(tau[R] - tau[L]) if R > L else 0.0

    out = pd.DataFrame([{
        "tau_star": tau_star,
        "max_abs_derivative": float(absd[idx]),
        "derivative_at_tau_star": float(dsucc[idx]),
        "half_peak_width_tau": width,
        "tau_min": float(tau.min()),
        "tau_max": float(tau.max()),
        "success_min": float(succ.min()),
        "success_max": float(succ.max()),
        "n_tau": int(len(tau)),
    }])
    out.to_csv(CSV_OUT, index=False)
    print(f"[OK] wrote {CSV_OUT}")

    # Plot derivative
    plt.figure()
    plt.plot(tau, dsucc)
    plt.axvline(tau_star, linestyle="--")
    plt.xlabel("tau")
    plt.ylabel("d(success)/d(tau)")
    plt.title("Phase transition sharpness (derivative peak)")
    plt.tight_layout()
    plt.savefig(PLOT_DER, dpi=200)
    print(f"[OK] saved {PLOT_DER}")

    # Plot success
    plt.figure()
    plt.plot(tau, succ)
    plt.axvline(tau_star, linestyle="--")
    plt.scatter([tau_star], [succ[idx]])
    plt.xlabel("tau")
    plt.ylabel("mean success")
    plt.title("Success vs tau with estimated critical tau*")
    plt.tight_layout()
    plt.savefig(PLOT_SUCC, dpi=200)
    print(f"[OK] saved {PLOT_SUCC}")

if __name__ == "__main__":
    main()
