import numpy as np
import heapq
import math
import torch
import matplotlib.pyplot as plt

from learned_heuristic import HeuristicNet

GRID_H, GRID_W = 40, 40

def make_grid():
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    grid[10:30, 15] = 1
    grid[5, 5:25] = 1
    grid[25:35, 25] = 1
    grid[15, 10:30] = 1
    return grid

def neighbors(pos, grid):
    x, y = pos
    res = []
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_H and 0 <= ny < GRID_W and grid[nx, ny] == 0:
            res.append((nx, ny))
    return res

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def reconstruct(came_from, start, goal):
    if goal not in came_from:
        return None
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    return path[::-1]

def astar_classic(grid, start, goal):
    open_pq = [(manhattan(start, goal), 0, start)]
    g = {start: 0}
    came_from = {start: start}
    expansions = 0

    while open_pq:
        f, cur_g, cur = heapq.heappop(open_pq)
        expansions += 1
        if cur == goal:
            break
        if cur_g > g[cur]:
            continue
        for nb in neighbors(cur, grid):
            tentative = cur_g + 1
            if nb not in g or tentative < g[nb]:
                g[nb] = tentative
                came_from[nb] = cur
                f_nb = tentative + manhattan(nb, goal)
                heapq.heappush(open_pq, (f_nb, tentative, nb))

    path = reconstruct(came_from, start, goal)
    return path, expansions

class LearnedHeuristicWrapper:
    def __init__(self, model_path="heuristic_net.pt", stats_path="planner_dataset_norm_stats.npz"):
        self.net = HeuristicNet(in_dim=4, hidden=64)
        self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.net.eval()
        stats = np.load(stats_path)
        self.mean = stats["mean"]
        self.std = stats["std"] + 1e-6

    def h(self, state, goal):
        x, y = state
        gx, gy = goal
        inp = np.array([x, y, gx, gy], dtype=np.float32)
        inp = (inp - self.mean) / self.std
        inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            val = self.net(inp).item()
        return max(0.0, float(val))  # cost-to-go δεν πρέπει να είναι αρνητικό

def astar_learned(grid, start, goal, learned_h):
    open_pq = [(learned_h.h(start, goal), 0, start)]
    g = {start: 0}
    came_from = {start: start}
    expansions = 0

    while open_pq:
        f, cur_g, cur = heapq.heappop(open_pq)
        expansions += 1
        if cur == goal:
            break
        if cur_g > g[cur]:
            continue
        for nb in neighbors(cur, grid):
            tentative = cur_g + 1
            if nb not in g or tentative < g[nb]:
                g[nb] = tentative
                came_from[nb] = cur
                f_nb = tentative + learned_h.h(nb, goal)
                heapq.heappush(open_pq, (f_nb, tentative, nb))

    path = reconstruct(came_from, start, goal)
    return path, expansions

def plot_paths(grid, start, goal, path_classic, path_learned):
    img = np.ones((GRID_H, GRID_W, 3))
    img[grid == 1] = [0, 0, 0]

    plt.figure(figsize=(6,6))
    plt.imshow(img, origin='upper')

    if path_classic:
        ys = [p[1] for p in path_classic]
        xs = [p[0] for p in path_classic]
        plt.plot(ys, xs, 'g-', label="A* classic")

    if path_learned:
        ys2 = [p[1] for p in path_learned]
        xs2 = [p[0] for p in path_learned]
        plt.plot(ys2, xs2, 'r--', label="A* learned heuristic")

    plt.scatter([start[1]], [start[0]], c='blue', label="Start")
    plt.scatter([goal[1]], [goal[0]], c='yellow', label="Goal")
    plt.legend()
    plt.title("A* Classic vs Learned Heuristic")
    plt.savefig("astar_learned_vs_classic.png", dpi=200)
    plt.show()

def main():
    grid = make_grid()
    start = (1, 1)
    goal = (35, 35)

    print("Running A* classic...")
    path_c, exp_c = astar_classic(grid, start, goal)
    print(f"classic: path length={len(path_c) if path_c else None}, expansions={exp_c}")

    print("Loading learned heuristic...")
    lh = LearnedHeuristicWrapper()

    print("Running A* with learned heuristic...")
    path_l, exp_l = astar_learned(grid, start, goal, lh)
    print(f"learned: path length={len(path_l) if path_l else None}, expansions={exp_l}")

    plot_paths(grid, start, goal, path_c, path_l)

if __name__ == "__main__":
    main()
