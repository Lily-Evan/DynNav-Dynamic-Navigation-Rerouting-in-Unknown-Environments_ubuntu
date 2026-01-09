import os
import time
import inspect
import numpy as np
import pandas as pd

import risk_weighted_planner as rwp
from ids_cost_shaping import synthetic_anomaly_stream, shape_lambda

OUT_CSV = "ids_cost_shaping_results.csv"

def _call_with_signature(func, kwargs):
    """
    Call func by matching only kwargs that exist in its signature.
    This makes it robust to minor API differences.
    """
    sig = inspect.signature(func)
    filt = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            filt[k] = v
    return func(**filt)

def make_world(n=80, obstacle_p=0.22, seed=0):
    """
    Build a simple synthetic occupancy + risk grid.
    occupancy: 0 free, 1 obstacle
    risk_grid: nonnegative float grid
    start, goal: free cells
    """
    rng = np.random.default_rng(seed)
    occ = (rng.random((n, n)) < obstacle_p).astype(np.uint8)

    # keep borders mostly free so start/goal likely feasible
    occ[0, :] = 0
    occ[:, 0] = 0
    occ[n-1, :] = 0
    occ[:, n-1] = 0

    # risk correlated field (smooth-ish)
    base = rng.random((n, n))
    risk = 0.2 + 0.8 * base  # in [0.2, 1.0]
    risk = risk.astype(np.float32)

    # choose start/goal
    start = (1, 1)
    goal = (n - 2, n - 2)
    occ[start] = 0
    occ[goal] = 0
    return occ, risk, start, goal

def normalize_path(path):
    """
    Path formats vary: list of tuples, list of lists, np arrays.
    Return list of (r,c) tuples.
    """
    if path is None:
        return None
    if isinstance(path, np.ndarray):
        path = path.tolist()
    out = []
    for p in path:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append((int(p[0]), int(p[1])))
    return out if len(out) > 0 else None

def extract_path(result):
    """
    Try to extract a path from various return types.
    - If result is list/np.ndarray -> assume it's the path.
    - If tuple -> try to find first list-like element that looks like a path.
    - If dict -> try 'path'
    """
    if result is None:
        return None

    if isinstance(result, dict):
        if "path" in result:
            return normalize_path(result["path"])
        # sometimes 'waypoints'
        if "waypoints" in result:
            return normalize_path(result["waypoints"])
        return None

    if isinstance(result, (list, np.ndarray)):
        return normalize_path(result)

    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, (list, np.ndarray)):
                p = normalize_path(item)
                if p is not None and len(p) >= 2:
                    return p
        # some APIs return (came_from, cost_so_far, ...)
        return None

    return None

def plan_real(occ, risk, start, goal, lam, risk_budget=None):
    """
    Use your real planner module. We try the adapter first (best),
    then astar_risk_weighted, then plan_under_risk_budget.

    Returns:
      path (list of (r,c)) or None
      meta dict with optional 'expansions' etc
    """
    meta = {}

    # Available kwargs for signature matching
    common = {
        # grids
        "occ_grid": occ,
        "occupancy": occ,
        "grid": occ,
        "occupancy_grid": occ,
        "risk_grid": risk,
        "risk": risk,
        # endpoints
        "start": start,
        "goal": goal,
        "start_rc": start,
        "goal_rc": goal,
        # risk knobs
        "lambda_risk": lam,
        "lam": lam,
        "lmbda": lam,
        "risk_lambda": lam,
        "risk_budget": risk_budget,
        "budget": risk_budget,
    }

    # 1) adapter (private but intended for this)
    if hasattr(rwp, "_adapter_plan_with_lambda"):
        try:
            res = _call_with_signature(rwp._adapter_plan_with_lambda, common)
            path = extract_path(res)
            if path is not None:
                meta["api_used"] = "_adapter_plan_with_lambda"
                return path, meta
        except Exception as e:
            meta["adapter_err"] = repr(e)

    # 2) astar_risk_weighted
    if hasattr(rwp, "astar_risk_weighted"):
        try:
            res = _call_with_signature(rwp.astar_risk_weighted, common)
            path = extract_path(res)
            if path is not None:
                meta["api_used"] = "astar_risk_weighted"
                # expansions sometimes returned in tuple; try to read if present
                if isinstance(res, tuple):
                    # heuristic: look for int-like
                    for item in res:
                        if isinstance(item, (int, np.integer)):
                            meta["expansions"] = int(item)
                            break
                return path, meta
        except Exception as e:
            meta["astar_err"] = repr(e)

    # 3) plan_under_risk_budget
    if hasattr(rwp, "plan_under_risk_budget"):
        try:
            if risk_budget is None:
                risk_budget = float(np.percentile(risk, 90)) * 200.0  # safe default
                common["risk_budget"] = risk_budget
                common["budget"] = risk_budget
            res = _call_with_signature(rwp.plan_under_risk_budget, common)
            path = extract_path(res)
            if path is not None:
                meta["api_used"] = "plan_under_risk_budget"
                return path, meta
        except Exception as e:
            meta["budget_err"] = repr(e)

    return None, meta

def path_metrics(path, risk):
    """
    Compute cost and risk using your helpers if possible,
    else fallback to simple grid metrics.
    """
    if path is None or len(path) < 2:
        return None

    # Use your internal functions if available
    try:
        L = float(rwp.path_length_l2(path))
    except Exception:
        # fallback: Manhattan-ish length
        L = float(sum(abs(path[i+1][0]-path[i][0]) + abs(path[i+1][1]-path[i][1]) for i in range(len(path)-1)))

    try:
        R = float(rwp.path_risk_sum(path, risk))
    except Exception:
        # fallback: sum risk on cells
        R = float(sum(float(risk[r, c]) for (r, c) in path))

    max_r = float(max(float(risk[r, c]) for (r, c) in path))
    return {"path_length": L, "path_risk_sum": R, "path_max_risk_cell": max_r}

def main():
    t0 = time.time()

    # Experiment settings
    seeds = list(range(10))
    attack_levels = [0.0, 0.4, 0.7, 0.9]   # 0.0 = no attack
    modes = ["baseline", "lambda_shaping"]

    # World settings
    n = 80
    obstacle_p = 0.22

    # Episode settings (replanning steps)
    T = 20  # keep modest; each step replans with new lambda
    attack_start, attack_end = 6, 15
    lambda0 = 1.0

    rows = []

    for seed in seeds:
        occ, risk, start, goal = make_world(n=n, obstacle_p=obstacle_p, seed=seed)

        for attack_level in attack_levels:
            scores = synthetic_anomaly_stream(
                T=T,
                attack_start=attack_start,
                attack_end=attack_end,
                base=0.05,
                attack_level=attack_level,
                noise=0.03,
                seed=seed
            )

            for mode in modes:
                total_cost = 0.0
                total_risk = 0.0
                max_risk_cell = 0.0
                ok_plans = 0
                expansions_sum = 0

                api_used = None
                last_err = None

                for t in range(T):
                    score = float(scores[t])
                    lam = lambda0 if mode == "baseline" else shape_lambda(lambda0, score, beta=2.5)

                    path, meta = plan_real(occ, risk, start, goal, lam, risk_budget=None)
                    api_used = api_used or meta.get("api_used", None)
                    last_err = meta.get("adapter_err") or meta.get("astar_err") or meta.get("budget_err") or last_err

                    if path is None:
                        continue

                    m = path_metrics(path, risk)
                    if m is None:
                        continue

                    ok_plans += 1
                    total_cost += m["path_length"]
                    total_risk += m["path_risk_sum"]
                    max_risk_cell = max(max_risk_cell, m["path_max_risk_cell"])

                    if "expansions" in meta:
                        expansions_sum += int(meta["expansions"])

                # Define episode success as: planned successfully in most steps
                episode_success = 1 if ok_plans >= int(0.8 * T) else 0

                rows.append({
                    "seed": seed,
                    "attack_level": attack_level,
                    "mode": mode,
                    "lambda0": lambda0,
                    "T": T,
                    "attack_start": attack_start,
                    "attack_end": attack_end,
                    "mean_score": float(np.mean(scores)),
                    "max_score": float(np.max(scores)),
                    "ok_plans": int(ok_plans),
                    "episode_success": int(episode_success),
                    "total_cost": float(total_cost),
                    "total_risk": float(total_risk),
                    "max_risk_cell": float(max_risk_cell),
                    "expansions_sum": int(expansions_sum),
                    "api_used": api_used or "none",
                    "last_err": last_err or "",
                    "used_toy_fallback": 0,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote {OUT_CSV} with {len(df)} rows")
    print(f"[TIME] {time.time()-t0:.2f}s")
    # quick check: did we actually use real API?
    used = df["api_used"].value_counts().to_dict()
    print("[INFO] api_used counts:", used)

if __name__ == "__main__":
    main()
