#!/usr/bin/env python3
import numpy as np
import random

class RRTStar:
    def __init__(self, start, goal, bounds=[0,10,0,10], step=0.5, radius=1.0, max_iter=2000):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds = bounds
        self.step = step
        self.radius = radius
        self.max_iter = max_iter
        self.nodes = [self.start]
        self.parents = {0: None}
        self.costs = {0: 0.0}

    def sample(self):
        x_min, x_max, y_min, y_max = self.bounds
        return np.array([
            random.uniform(x_min, x_max),
            random.uniform(y_min, y_max)
        ])

    def nearest(self, sample):
        dists = [np.linalg.norm(n - sample) for n in self.nodes]
        return int(np.argmin(dists))

    def steer(self, nearest_node, sample):
        direction = sample - nearest_node
        dist = np.linalg.norm(direction)
        if dist < self.step:
            return sample
        return nearest_node + (direction/dist)*self.step

    def plan(self):
        for i in range(self.max_iter):
            s = self.sample()
            idx = self.nearest(s)
            nearest_node = self.nodes[idx]
            new_node = self.steer(nearest_node, s)

            self.nodes.append(new_node)
            new_id = len(self.nodes) - 1
            self.parents[new_id] = idx
            self.costs[new_id] = self.costs[idx] + np.linalg.norm(new_node - nearest_node)

            if np.linalg.norm(new_node - self.goal) < 0.5:
                return self.extract_path(new_id)

        return None

    def extract_path(self, last_id):
        path = []
        while last_id is not None:
            path.append(self.nodes[last_id])
            last_id = self.parents[last_id]
        return path[::-1]
