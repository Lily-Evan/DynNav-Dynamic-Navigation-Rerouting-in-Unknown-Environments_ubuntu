"""
IG-based Planning Demo

Διαβάζει τον επόμενο στόχο από ig_next_goal.csv
και τρέχει απλούς PRM/RRT planners προς αυτόν τον στόχο.
Δεν πειράζει το advanced_planners.py.
"""

import math
import random
import pandas as pd

# =========================
# 1. Φόρτωση IG goal από CSV
# =========================

def load_ig_goal(csv_path: str = "ig_next_goal.csv"):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"No rows found in {csv_path}")
    row = df.iloc[0]
    goal_x = float(row["goal_x"])
    goal_y = float(row["goal_y"])
    return goal_x, goal_y

# =========================
# 2. Απλός PRM demo (ευθύγραμμη διαδρομή)
# =========================

def prm_plan(start, goal, steps=20):
    print(f"[PRM] Planning from {start} to {goal}")
    sx, sy = start
    gx, gy = goal
    path = []
    for i in range(steps + 1):
        t = i / steps
        x = sx + (gx - sx) * t
        y = sy + (gy - sy) * t
        path.append((x, y))
    return path

# =========================
# 3. Απλός RRT demo
# =========================

def rrt_plan(start, goal, iterations=200):
    print(f"[RRT] Planning from {start} to {goal}")

    nodes = [start]
    parents = {start: None}

    for i in range(iterations):
        rx = random.uniform(0, 50)
        ry = random.uniform(0, 50)

        nearest = min(nodes, key=lambda p: (p[0] - rx) ** 2 + (p[1] - ry) ** 2)

        new = (
            nearest[0] + (rx - nearest[0]) * 0.3,
            nearest[1] + (ry - nearest[1]) * 0.3,
        )

        nodes.append(new)
        parents[new] = nearest

        if math.dist(new, goal) < 1.5:
            print(f"[RRT] Goal reached at iteration {i}")
            path = [new]
            while path[-1] != start:
                path.append(parents[path[-1]])
            return path[::-1]

    print("[RRT] Goal not reached within iteration limit")
    return nodes

# =========================
# 4. Main
# =========================

if __name__ == "__main__":
    print("=== IG-based PRM/RRT demo ===")

    # start σημείο (μπορείς να το αλλάξεις)
    start = (0.0, 0.0)

    # IG goal από το info_gain_planner.py
    goal_x, goal_y = load_ig_goal("ig_next_goal.csv")
    goal = (goal_x, goal_y)

    print(f"[IG] Using goal from ig_next_goal.csv: ({goal_x:.2f}, {goal_y:.2f})")

    prm_path = prm_plan(start, goal)
    print(f"[PRM] path length (points): {len(prm_path)}")

    rrt_path = rrt_plan(start, goal)
    print(f"[RRT] path length (points): {len(rrt_path)}")
