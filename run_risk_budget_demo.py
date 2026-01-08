import numpy as np
from risk_weighted_planner import plan_under_risk_budget, _adapter_plan_with_lambda

def main():
    # Dummy risk grid: low risk everywhere, high risk stripe in the middle
    W, H = 60, 40
    risk = np.ones((H, W), dtype=float) * 0.1
    risk[:, 28:32] = 2.0  # risky vertical band

    # IMPORTANT: start/goal are (x, y)
    start = (5, 20)
    goal  = (55, 20)

    # Start with a large budget to verify it works, then tune down
    B = 2000.0

    best = plan_under_risk_budget(
        start=start,
        goal=goal,
        risk_grid=risk,
        B=B,
        plan_with_lambda_fn=_adapter_plan_with_lambda,
        lambda0=1.0,
        eta=0.5,
        max_dual_iters=15,
    )

    if best is None:
        print("No feasible path found.")
        return

    print("FOUND feasible path")
    print("L =", best["L"])
    print("R =", best["R"])
    print("B =", best["B"])
    print("lambda_final =", best["lambda"])
    print("iters =", best["iters"])
    print("path_len_points =", len(best["path"]))
    print("first 5 points:", best["path"][:5])

if __name__ == "__main__":
    main()
