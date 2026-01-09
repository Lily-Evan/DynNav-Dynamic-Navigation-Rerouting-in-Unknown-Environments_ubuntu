import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq

# Ποιον αλγόριθμο να κάνουμε plot ("bfs","dijkstra","greedy","astar")
#PLANNER_TO_PLOT = "astar"


#PLANNER_TO_PLOT = "bfs"
# ή
PLANNER_TO_PLOT = "dijkstra"
# ή
#PLANNER_TO_PLOT = "greedy"



GRID_H, GRID_W = 40, 40

def make_grid():
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    # μερικά εμπόδια για να έχει ενδιαφέρον
    grid[10:30, 15] = 1
    grid[5, 5:25] = 1
    grid[25:35, 25] = 1
    grid[15, 10:30] = 1
    return grid

def get_neighbors(pos, grid):
    h, w = grid.shape
    x, y = pos
    nbrs = []
    # 4-γειτονιά (πάνω, κάτω, αριστερά, δεξιά)
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 0:
            nbrs.append((nx, ny))
    return nbrs

def reconstruct(came_from, start, goal):
    if goal not in came_from:
        return None
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path

# ---------------- BFS ----------------
def bfs(grid, start, goal):
    q = deque([start])
    came_from = {start: start}
    visited = {start}
    expansions = 0
    while q:
        cur = q.popleft()
        expansions += 1
        if cur == goal:
            break
        for nb in get_neighbors(cur, grid):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = cur
                q.append(nb)
    return reconstruct(came_from, start, goal), expansions

# --------------- Dijkstra ---------------
def dijkstra(grid, start, goal):
    pq = [(0, start)]
    came_from = {start: start}
    cost = {start: 0}
    expansions = 0
    while pq:
        cur_cost, cur = heapq.heappop(pq)
        expansions += 1
        if cur == goal:
            break
        if cur_cost > cost[cur]:
            continue
        for nb in get_neighbors(cur, grid):
            new_c = cur_cost + 1  # κόστος edge = 1
            if nb not in cost or new_c < cost[nb]:
                cost[nb] = new_c
                came_from[nb] = cur
                heapq.heappush(pq, (new_c, nb))
    return reconstruct(came_from, start, goal), expansions

# --------------- Greedy Best-First ---------------
def heuristic(a, b):
    # Manhattan distance
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def greedy(grid, start, goal):
    pq = [(heuristic(start, goal), start)]
    came_from = {start: start}
    visited = {start}
    expansions = 0
    while pq:
        _, cur = heapq.heappop(pq)
        expansions += 1
        if cur == goal:
            break
        for nb in get_neighbors(cur, grid):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = cur
                heapq.heappush(pq, (heuristic(nb, goal), nb))
    return reconstruct(came_from, start, goal), expansions

# --------------- A* ---------------
def astar(grid, start, goal):
    pq = [(heuristic(start, goal), 0, start)]
    came_from = {start: start}
    g = {start: 0}
    expansions = 0
    while pq:
        f, cur_g, cur = heapq.heappop(pq)
        expansions += 1
        if cur == goal:
            break
        if cur_g > g[cur]:
            continue
        for nb in get_neighbors(cur, grid):
            tentative = cur_g + 1
            if nb not in g or tentative < g[nb]:
                g[nb] = tentative
                came_from[nb] = cur
                f_nb = tentative + heuristic(nb, goal)
                heapq.heappush(pq, (f_nb, tentative, nb))
    return reconstruct(came_from, start, goal), expansions

# --------------- Plot ---------------
def plot_grid_and_path(grid, path, start, goal, title):
    h, w = grid.shape
    img = np.ones((h, w, 3))
    # εμπόδια μαύρα
    img[grid == 1] = [0, 0, 0]
    plt.figure(figsize=(6,6))
    plt.imshow(img, origin='upper')
    if path:
        ys = [p[1] for p in path]
        xs = [p[0] for p in path]
        plt.plot(ys, xs, linewidth=2)
    # start πράσινο, goal κόκκινο
    plt.scatter([start[1]], [start[0]], c='g')
    plt.scatter([goal[1]], [goal[0]], c='r')
    plt.title(title)
    plt.grid(False)
    plt.savefig("planner_path.png", dpi=200)
    plt.show()

def run():
    grid = make_grid()
    start = (1, 1)
    goal = (GRID_H-2, GRID_W-2)

    algos = {
        "bfs": bfs,
        "dijkstra": dijkstra,
        "greedy": greedy,
        "astar": astar,
    }

    print("Start:", start, "Goal:", goal)
    for name, func in algos.items():
        path, expansions = func(grid, start, goal)
        plen = len(path) if path is not None else None
        print(f"{name:9s} -> path length = {plen}, expanded nodes = {expansions}")

    # Plot για τον επιλεγμένο planner
    to_plot = PLANNER_TO_PLOT
    if to_plot in algos:
        path, _ = algos[to_plot](grid, start, goal)
        plot_grid_and_path(grid, path, start, goal, f"{to_plot} path")
    else:
        print("Unknown planner to plot:", to_plot)

if __name__ == "__main__":
    run()
