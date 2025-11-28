import numpy as np

class RRT:
    def __init__(self, step=5, max_iter=2000):
        self.step = step
        self.max_iter = max_iter

    def distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def steer(self, from_node, to_node):
        from_pt = np.array(from_node)
        to_pt = np.array(to_node)

        direction = to_pt - from_pt
        dist = np.linalg.norm(direction)
        direction = direction / (dist + 1e-6)

        return tuple((from_pt + direction*self.step).astype(int))

    def nearest(self, nodes, rnd):
        dists = [self.distance(n, rnd) for n in nodes]
        return nodes[np.argmin(dists)]

    def is_collision_free(self, occ, p):
        return occ[p[1], p[0]] == 0

    def plan(self, occ_grid, start, goal):
        nodes = [start]
        parent = {start: None}

        for _ in range(self.max_iter):
            rnd = (np.random.randint(0, occ_grid.shape[1]), 
                   np.random.randint(0, occ_grid.shape[0]))

            nearest_node = self.nearest(nodes, rnd)
            new_node = self.steer(nearest_node, rnd)

            if not self.is_collision_free(occ_grid, new_node):
                continue

            nodes.append(new_node)
            parent[new_node] = nearest_node

            if self.distance(new_node, goal) < self.step:
                parent[goal] = new_node
                break

        # reconstruct path
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        return path[::-1]

