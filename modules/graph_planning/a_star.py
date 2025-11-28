#!/usr/bin/env python3
import heapq
import numpy as np

class AStar:
    def __init__(self, grid):
        self.grid = grid  # 0 free, 1 obstacle
        self.h = grid.shape[0]
        self.w = grid.shape[1]

    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def neighbors(self, node):
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        result = []
        for dx,dy in dirs:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.w and 0 <= ny < self.h and self.grid[ny][nx] == 0:
                result.append((nx,ny))
        return result

    def plan(self, start, goal):
        pq = []
        heapq.heappush(pq, (0, start))
        came = {start: None}
        cost = {start: 0}

        while pq:
            _, current = heapq.heappop(pq)

            if current == goal:
                return self.extract_path(came, current)

            for n in self.neighbors(current):
                new_cost = cost[current] + 1
                if n not in cost or new_cost < cost[n]:
                    cost[n] = new_cost
                    priority = new_cost + self.heuristic(n, goal)
                    heapq.heappush(pq, (priority, n))
                    came[n] = current
        return None

    def extract_path(self, came, current):
        path = []
        while current:
            path.append(current)
            current = came[current]
        return path[::-1]
