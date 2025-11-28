#!/usr/bin/env python3
import numpy as np
from sklearn.neighbors import NearestNeighbors

class PRMPlanner:
    def __init__(self, n_samples=500, k=10, bounds=[0,10,0,10]):
        self.n_samples = n_samples
        self.k = k
        self.bounds = bounds

    def sample_points(self):
        x_min, x_max, y_min, y_max = self.bounds
        X = np.random.uniform(x_min, x_max, self.n_samples)
        Y = np.random.uniform(y_min, y_max, self.n_samples)
        return np.vstack([X, Y]).T

    def build_graph(self, points):
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(points)
        distances, indices = nbrs.kneighbors(points)
        return indices

    def plan(self, start, goal):
        pts = self.sample_points()
        graph = self.build_graph(pts)

        # Connect start + goal
        pts = np.vstack([start, goal, pts])
        start_id = 0
        goal_id = 1

        # Simple A* over PRM graph (pseudo)
        from heapq import heappush, heappop
        N = len(pts)
        visited = set()
        pq = []
        heappush(pq, (0, start_id, []))

        while pq:
            cost, node, path = heappop(pq)

            if node in visited:
                continue
            visited.add(node)

            new_path = path + [pts[node]]

            if node == goal_id:
                return new_path

            neighbors = [i for i in range(2, N)]  # naive fully connected
            for nb in neighbors:
                dist = np.linalg.norm(pts[node] - pts[nb])
                heappush(pq, (cost + dist, nb, new_path))

        return None
