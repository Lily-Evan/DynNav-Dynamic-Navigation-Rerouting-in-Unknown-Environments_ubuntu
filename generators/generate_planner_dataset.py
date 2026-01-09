import numpy as np
import heapq
import random

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

def dijkstra_from_goal(grid, goal):
    INF = 1e9
    dist = np.full((GRID_H, GRID_W), INF, dtype=float)
    gx, gy = goal
    if grid[gx, gy] == 1:
        return dist  # κακό goal, όλα INF

    dist[gx, gy] = 0.0
    pq = [(0.0, goal)]
    while pq:
        d, (x, y) = heapq.heappop(pq)
        if d > dist[x, y]:
            continue
        for nx, ny in neighbors((x, y), grid):
            nd = d + 1.0
            if nd < dist[nx, ny]:
                dist[nx, ny] = nd
                heapq.heappush(pq, (nd, (nx, ny)))
    return dist

def main():
    grid = make_grid()
    X = []
    y = []

    num_goals = 80  # πόσα goals θα δειγματοληπτήσουμε
    for i in range(num_goals):
        gx = random.randint(0, GRID_H-1)
        gy = random.randint(0, GRID_W-1)
        if grid[gx, gy] == 1:
            continue
        goal = (gx, gy)
        print(f"[{i+1}/{num_goals}] Dijkstra from goal {goal}")
        dist = dijkstra_from_goal(grid, goal)

        for x in range(GRID_H):
            for y_ in range(GRID_W):
                if grid[x, y_] == 1:
                    continue
                d = dist[x, y_]
                if d >= 1e9:
                    continue  # μη προσβάσιμο
                X.append([x, y_, gx, gy])
                y.append(d)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print("Dataset size:", X.shape, y.shape)
    np.savez("planner_dataset.npz", X=X, y=y)
    print("Saved planner_dataset.npz")

if __name__ == "__main__":
    main()
