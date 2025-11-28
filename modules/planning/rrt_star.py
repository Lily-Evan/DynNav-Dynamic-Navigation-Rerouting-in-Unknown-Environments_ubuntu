import numpy as np

class RRTStar:
    def __init__(self, step=5, radius=20, max_iter=2000):
        self.step = step
        self.radius = radius
        self.max_iter = max_iter

    def dist(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def steer(self, a, b):
        a = np.array(a)
        b = np.array(b)
        d = b - a
        d /= np.linalg.norm(d) + 1e-6
        return tuple((a + d*self.step).astype(int))

    def near(self, nodes, node):
        return [n for n in nodes if self.dist(n, node) < self.radius]

    def plan(self, occ, start, goal):
        nodes = [start]
        parent = {start: None}
        cost = {start: 0}

        for _ in range(self.max_iter):
            rnd = (np.random.randint(0, occ.shape[1]), 
                   np.random.randint(0, occ.shape[0]))

            nearest = min(nodes, key=lambda n: self.dist(n, rnd))
            new = self.steer(nearest, rnd)

            if occ[new[1], new[0]] != 0:
                continue

            near_nodes = self.near(nodes, new)
            best = nearest
            best_cost = cost[nearest] + self.dist(nearest, new)

            for n in near_nodes:
                c = cost[n] + self.dist(n, new)
                if c < best_cost:
                    best = n
                    best_cost = c

            nodes.append(new)
            parent[new] = best
            cost[new] = best_cost

            for n in near_nodes:
                if cost[n] > cost[new] + self.dist(n, new):
                    parent[n] = new
                    cost[n] = cost[new] + self.dist(n, new)

            if self.dist(new, goal) < self.step:
                parent[goal] = new
                cost[goal] = cost[new] + self.dist(new, goal)
                break

        # reconstruct path
        path = []
        cur = goal
        while cur:
            path.append(cur)
            cur = parent[cur]
        return path[::-1]
