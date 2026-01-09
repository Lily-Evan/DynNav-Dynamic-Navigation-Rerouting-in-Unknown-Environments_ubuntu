import numpy as np
import matplotlib.pyplot as plt
import math
import random

GRID_H, GRID_W = 100, 100

# -----------------------------
# Utility functions
# -----------------------------
def make_grid():
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    # μερικά εμπόδια για να έχει ενδιαφέρον
    grid[20:80, 40] = 1
    grid[50, 10:70] = 1
    grid[30:70, 70] = 1
    return grid

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def collision_free(p1, p2, grid):
    x1, y1 = p1
    x2, y2 = p2
    steps = int(max(1, np.hypot(x2 - x1, y2 - y1)))
    for i in range(steps+1):
        t = i / steps
        x = int(round(x1 + t * (x2 - x1)))
        y = int(round(y1 + t * (y2 - y1)))
        if x < 0 or x >= GRID_H or y < 0 or y >= GRID_W:
            return False
        if grid[x, y] == 1:
            return False
    return True

# -----------------------------
# Baseline RRT*
# -----------------------------
def rrt_star(grid, start, goal, step=4, radius=15, max_iters=4000):
    nodes = [start]
    parents = {start: None}
    costs = {start: 0.0}
    best_goal = None
    best_cost = float("inf")
    cost_history = []

    for it in range(max_iters):
        # uniform sampling
        if random.random() < 0.05:
            rnd = goal
        else:
            rnd = (random.randint(0, GRID_H-1), random.randint(0, GRID_W-1))

        nearest = min(nodes, key=lambda n: dist(n, rnd))

        theta = math.atan2(rnd[1] - nearest[1], rnd[0] - nearest[0])
        new = (int(nearest[0] + step * math.cos(theta)),
               int(nearest[1] + step * math.sin(theta)))

        if new[0] < 0 or new[0] >= GRID_H or new[1] < 0 or new[1] >= GRID_W:
            continue
        if not collision_free(nearest, new, grid):
            continue

        nodes.append(new)

        # neighbors με κόστος
        nbrs = [n for n in nodes if n in costs and dist(n, new) < radius]

        # choose best parent
        best_parent = nearest
        best_new_cost = costs[nearest] + dist(nearest, new)
        for nb in nbrs:
            if nb == new:
                continue
            if collision_free(nb, new, grid):
                c = costs[nb] + dist(nb, new)
                if c < best_new_cost:
                    best_new_cost = c
                    best_parent = nb

        parents[new] = best_parent
        costs[new] = best_new_cost

        # rewiring
        for nb in nbrs:
            if nb == new:
                continue
            if collision_free(new, nb, grid):
                c = costs[new] + dist(new, nb)
                if c < costs.get(nb, float("inf")):
                    parents[nb] = new
                    costs[nb] = c

        # check goal
        if dist(new, goal) < 5 and collision_free(new, goal, grid):
            goal_cost = costs[new] + dist(new, goal)
            if goal_cost < best_cost:
                best_cost = goal_cost
                best_goal = new

        cost_history.append(best_cost)

    path = None
    if best_goal is not None:
        parents[goal] = best_goal
        costs[goal] = best_cost
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parents[cur]
        path.reverse()

    return path, cost_history

# -----------------------------
# Informed RRT*
# -----------------------------
def sample_unit_ball_2d():
    # uniform στο δίσκο
    u = random.random()
    r = math.sqrt(u)
    theta = random.random() * 2 * math.pi
    return np.array([r * math.cos(theta), r * math.sin(theta)])

def informed_sample(start, goal, c_best):
    """
    Sample μέσα στο prolate ellipsoid που περιέχει όλες τις
    feasible λύσεις με cost <= c_best.
    """
    # c_min = straight-line distance
    c_min = dist(start, goal)
    if c_best == float("inf") or c_best < c_min:
        # δεν έχουμε λύση ακόμη ή κάτι περίεργο -> uniform sample
        return None

    start_np = np.array([start[0], start[1]], dtype=float)
    goal_np = np.array([goal[0], goal[1]], dtype=float)

    center = (start_np + goal_np) / 2.0

    # major axis direction
    e1 = (goal_np - start_np) / c_min
    # 2D perpendicular
    e2 = np.array([-e1[1], e1[0]])

    C = np.column_stack((e1, e2))  # rotation matrix 2x2

    a = c_best / 2.0
    b_sq = (c_best**2 - c_min**2) / 4.0
    if b_sq < 1e-9:
        b = 1e-4
    else:
        b = math.sqrt(b_sq)

    L = np.diag([a, b])

    x_ball = sample_unit_ball_2d()
    x_rand = C @ (L @ x_ball) + center
    return (int(round(x_rand[0])), int(round(x_rand[1])))

def informed_rrt_star(grid, start, goal, step=4, radius=15, max_iters=4000):
    nodes = [start]
    parents = {start: None}
    costs = {start: 0.0}
    best_goal = None
    best_cost = float("inf")
    cost_history = []

    for it in range(max_iters):
        # Informed sampling: αν έχουμε βρει ήδη λύση, δειγμάτισε στο ellipsoid
        rnd = None
        if best_goal is not None:
            sample = informed_sample(start, goal, best_cost)
            if sample is not None:
                rnd = sample

        if rnd is None:
            # fallback: uniform sampling
            if random.random() < 0.05:
                rnd = goal
            else:
                rnd = (random.randint(0, GRID_H-1), random.randint(0, GRID_W-1))

        nearest = min(nodes, key=lambda n: dist(n, rnd))
        theta = math.atan2(rnd[1] - nearest[1], rnd[0] - nearest[0])
        new = (int(nearest[0] + step * math.cos(theta)),
               int(nearest[1] + step * math.sin(theta)))

        if new[0] < 0 or new[0] >= GRID_H or new[1] < 0 or new[1] >= GRID_W:
            continue
        if not collision_free(nearest, new, grid):
            continue

        nodes.append(new)

        nbrs = [n for n in nodes if n in costs and dist(n, new) < radius]

        best_parent = nearest
        best_new_cost = costs[nearest] + dist(nearest, new)
        for nb in nbrs:
            if nb == new:
                continue
            if collision_free(nb, new, grid):
                c = costs[nb] + dist(nb, new)
                if c < best_new_cost:
                    best_new_cost = c
                    best_parent = nb

        parents[new] = best_parent
        costs[new] = best_new_cost

        # rewiring
        for nb in nbrs:
            if nb == new:
                continue
            if collision_free(new, nb, grid):
                c = costs[new] + dist(new, nb)
                if c < costs.get(nb, float("inf")):
                    parents[nb] = new
                    costs[nb] = c

        # check goal
        if dist(new, goal) < 5 and collision_free(new, goal, grid):
            goal_cost = costs[new] + dist(new, goal)
            if goal_cost < best_cost:
                best_cost = goal_cost
                best_goal = new

        cost_history.append(best_cost)

    path = None
    if best_goal is not None:
        parents[goal] = best_goal
        costs[goal] = best_cost
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parents[cur]
        path.reverse()

    return path, cost_history

# -----------------------------
# Plot helpers
# -----------------------------
def plot_paths(grid, start, goal, path_rrt, path_inf):
    img = np.ones((GRID_H, GRID_W, 3))
    img[grid == 1] = [0, 0, 0]

    plt.figure(figsize=(6, 6))
    plt.imshow(img, origin='upper')

    if path_rrt:
        ys = [p[1] for p in path_rrt]
        xs = [p[0] for p in path_rrt]
        plt.plot(ys, xs, 'g-', label="RRT*")

    if path_inf:
        ys2 = [p[1] for p in path_inf]
        xs2 = [p[0] for p in path_inf]
        plt.plot(ys2, xs2, 'r--', label="Informed-RRT*")

    plt.scatter([start[1]], [start[0]], c='blue', label="Start")
    plt.scatter([goal[1]], [goal[0]], c='yellow', label="Goal")
    plt.title("RRT* vs Informed-RRT*")
    plt.legend()
    plt.savefig("informed_rrt_star_paths.png", dpi=200)
    plt.show()

def plot_costs(cost_rrt, cost_inf):
    iters = np.arange(len(cost_rrt))
    iters2 = np.arange(len(cost_inf))
    plt.figure()
    plt.plot(iters, cost_rrt, label="RRT* best cost")
    plt.plot(iters2, cost_inf, label="Informed-RRT* best cost")
    plt.xlabel("Iteration")
    plt.ylabel("Best path cost so far")
    plt.title("Cost convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("informed_rrt_star_costs.png", dpi=200)
    plt.show()

# -----------------------------
# MAIN
# -----------------------------
def main():
    grid = make_grid()
    start = (5, 5)
    goal = (90, 90)

    print("Running RRT* ...")
    path_rrt, cost_rrt = rrt_star(grid, start, goal)
    if path_rrt:
        print("RRT* path length:", len(path_rrt))
    else:
        print("RRT* failed to find path")

    print("Running Informed-RRT* ...")
    path_inf, cost_inf = informed_rrt_star(grid, start, goal)
    if path_inf:
        print("Informed-RRT* path length:", len(path_inf))
    else:
        print("Informed-RRT* failed to find path")

    plot_paths(grid, start, goal, path_rrt, path_inf)
    plot_costs(cost_rrt, cost_inf)

if __name__ == "__main__":
    main()
