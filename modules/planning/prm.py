import numpy as np
from sklearn.neighbors import NearestNeighbors

class PRM:
    def __init__(self, n_samples=300, k=10):
        self.n = n_samples
        self.k = k

    def sample_free(self, occ_grid):
        free_cells = np.argwhere(occ_grid == 0)
        idx = np.random.choice(len(free_cells), self.n)
        return free_cells[idx]

    def build_roadmap(self, points):
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(points)
        dists, inds = nbrs.kneighbors(points)

        edges = []
        for i, neighbors in enumerate(inds):
            for j in neighbors[1:]:
                edges.append((tuple(points[i]), tuple(points[j])))

        return edges

    def find_path(self, edges, start, goal):
        # simple BFS on roadmap
        graph = {}
        for a, b in edges:
            graph.setdefault(a, []).append(b)
            graph.setdefault(b, []).append(a)

        visited = set([start])
        q = [(start, [start])]

        while q:
            node, path = q.pop(0)
            if node == goal:
                return path

            for neigh in graph.get(node, []):
                if neigh not in visited:
                    visited.add(neigh)
                    q.append((neigh, path + [neigh]))

        return None
