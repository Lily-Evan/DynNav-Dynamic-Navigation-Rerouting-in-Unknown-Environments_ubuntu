import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
import math

GRID_H, GRID_W = 100, 100

# -------------------------
# Utility functions
# -------------------------
def make_grid():
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

    grid[20:80, 40] = 1
    grid[50, 10:70] = 1
    grid[30:70, 70] = 1
    return grid

def collision_free(p1, p2, grid):
    x1, y1 = p1
    x2, y2 = p2
    steps = int(np.hypot(x2 - x1, y2 - y1))
    for i in range(steps + 1):
        t = i / max(1, steps)
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        if x < 0 or x >= GRID_H or y < 0 or y >= GRID_W or grid[x, y] == 1:
            return False
    return True

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# -----------------------------------------
# PRM
# -----------------------------------------
def prm(grid, start, goal, n_samples=500, k=10):
    samples = [start, goal]
    samples.extend([(random.randint(0, GRID_H - 1), random.randint(0, GRID_W - 1))
                    for _ in range(n_samples)])

    samples = [s for s in samples if grid[s[0], s[1]] == 0]

    graph = {i: [] for i in range(len(samples))}

    for i, p in enumerate(samples):
        dists = [(dist(p, samples[j]), j) for j in range(len(samples)) if j != i]
        dists.sort()
        for d, j in dists[:k]:
            if collision_free(p, samples[j], grid):
                graph[i].append(j)
                graph[j].append(i)

    def dijkstra(start_idx, goal_idx):
        pq = [(0, start_idx)]
        cost = {start_idx: 0}
        parent = {start_idx: None}
        while pq:
            c, u = heapq.heappop(pq)
            if u == goal_idx:
                break
            for v in graph[u]:
                nc = c + dist(samples[u], samples[v])
                if v not in cost or nc < cost[v]:
                    cost[v] = nc
                    parent[v] = u
                    heapq.heappush(pq, (nc, v))
        if goal_idx not in parent:
            return None
        path = []
        cur = goal_idx
        while cur is not None:
            path.append(samples[cur])
            cur = parent[cur]
        return path[::-1]

    start_idx = samples.index(start)
    goal_idx = samples.index(goal)
    return dijkstra(start_idx, goal_idx)

# -----------------------------------------
# RRT
# -----------------------------------------
def rrt(grid, start, goal, step=4, max_iters=4000):
    nodes = [start]
    parents = {start: None}
    for it in range(max_iters):
        if random.random() < 0.05:
            rnd = goal
        else:
            rnd = (random.randint(0, GRID_H - 1), random.randint(0, GRID_W - 1))

        nearest = min(nodes, key=lambda n: dist(n, rnd))

        theta = math.atan2(rnd[1] - nearest[1], rnd[0] - nearest[0])
        new = (int(nearest[0] + step * math.cos(theta)),
               int(nearest[1] + step * math.sin(theta)))

        if new[0] < 0 or new[0] >= GRID_H or new[1] < 0 or new[1] >= GRID_W:
            continue

        if collision_free(nearest, new, grid):
            nodes.append(new)
            parents[new] = nearest
            if dist(new, goal) < 5:
                parents[goal] = new
                break

    if goal not in parents:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    return path[::-1]

# -----------------------------------------
# RRT*
# -----------------------------------------
def rrt_star(grid, start, goal, step=4, radius=15, max_iters=4000):
    nodes = [start]
    parents = {start: None}
    costs = {start: 0}

    for it in range(max_iters):
        # sample
        if random.random() < 0.05:
            rnd = goal
        else:
            rnd = (random.randint(0, GRID_H - 1), random.randint(0, GRID_W - 1))

        # nearest
        nearest = min(nodes, key=lambda n: dist(n, rnd))

        # steer
        theta = math.atan2(rnd[1] - nearest[1], rnd[0] - nearest[0])
        new = (int(nearest[0] + step * math.cos(theta)),
               int(nearest[1] + step * math.sin(theta)))

        # bounds check
        if new[0] < 0 or new[0] >= GRID_H or new[1] < 0 or new[1] >= GRID_W:
            continue
        if not collision_free(nearest, new, grid):
            continue

        # προσωρινά βάζουμε το new στη λίστα nodes
        nodes.append(new)

        # neighbors μέσα σε ακτίνα που έχουν ΗΔΗ κόστος (είναι γνωστοί κόμβοι)
        nbrs = [n for n in nodes if n in costs and dist(n, new) < radius]

        # επιλέγουμε best parent
        best_parent = nearest
        best_cost = costs[nearest] + dist(nearest, new)

        for nb in nbrs:
            if nb == new:
                continue
            if collision_free(nb, new, grid):
                c = costs[nb] + dist(nb, new)
                if c < best_cost:
                    best_cost = c
                    best_parent = nb

        parents[new] = best_parent
        costs[new] = best_cost

        # rewiring γειτόνων
        for nb in nbrs:
            if nb == new:
                continue
            if collision_free(new, nb, grid):
                c = costs[new] + dist(new, nb)
                if c < costs.get(nb, float("inf")):
                    parents[nb] = new
                    costs[nb] = c

        # έλεγχος goal
        if dist(new, goal) < 5 and collision_free(new, goal, grid):
            parents[goal] = new
            costs[goal] = costs[new] + dist(new, goal)
            break

    if goal not in parents:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    return path[::-1]

# -----------------------------------------
# FMT* (simplified)
# -----------------------------------------
def fmt_star(grid, start, goal, n_samples=1000, radius=20):
    samples = [(random.randint(0, GRID_H-1), random.randint(0, GRID_W-1)) for _ in range(n_samples)]
    samples = [s for s in samples if grid[s[0], s[1]] == 0]
    samples = [start] + samples + [goal]

    open_set = {start}
    closed_set = set()
    costs = {start: 0}
    parents = {start: None}

    while open_set:
        cur = min(open_set, key=lambda n: costs[n])
        open_set.remove(cur)
        closed_set.add(cur)

        if cur == goal:
            break

        nbrs = [s for s in samples if dist(s, cur) < radius and s not in closed_set]
        for nb in nbrs:
            if collision_free(cur, nb, grid):
                new_cost = costs[cur] + dist(cur, nb)
                if new_cost < costs.get(nb, float("inf")):
                    costs[nb] = new_cost
                    parents[nb] = cur
                    open_set.add(nb)

    if goal not in parents:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    return path[::-1]

# -----------------------------------------
# BIT* (very simplified batch version)
# -----------------------------------------
def bit_star(grid, start, goal, batches=5, samples_per_batch=300, radius=20):
    nodes = [start]
    parents = {start: None}
    costs = {start: 0}

    for batch in range(batches):
        batch_samples = [
            (random.randint(0, GRID_H-1), random.randint(0, GRID_W-1))
            for _ in range(samples_per_batch)
        ]
        batch_samples = [s for s in batch_samples if grid[s[0], s[1]] == 0]

        for s in batch_samples:
            nearest = min(nodes, key=lambda n: dist(n, s))
            if dist(nearest, s) < radius and collision_free(nearest, s, grid):
                nodes.append(s)
                parents[s] = nearest
                costs[s] = costs[nearest] + dist(nearest, s)

        if dist(nodes[-1], goal) < 5:
            parents[goal] = nodes[-1]
            break

    if goal not in parents:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    return path[::-1]

# -----------------------------------------
# Plotting
# -----------------------------------------
def plot(grid, path, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray_r")
    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        plt.plot(xs, ys, c="red")
    plt.title(title)
    plt.show()

# -----------------------------------------
# MAIN
# -----------------------------------------
def run():
    grid = make_grid()
    start = (5, 5)
    goal = (90, 90)

    planners = {
        "PRM": lambda: prm(grid, start, goal),
        "RRT": lambda: rrt(grid, start, goal),
        "RRT*": lambda: rrt_star(grid, start, goal),
        "FMT*": lambda: fmt_star(grid, start, goal),
        "BIT*": lambda: bit_star(grid, start, goal),
    }

    for name, func in planners.items():
        print(f"\nRunning {name} ...")
        path = func()
        if path:
            print(f"{name} path length = {len(path)}")
        else:
            print(f"{name} FAILED to find path.")
        plot(grid, path, name)

if __name__ == "__main__":
    run()
