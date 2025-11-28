import heapq
import numpy as np

class AStar:
    def __init__(self):
        pass

    def h(self, a, b):
        return np.linalg.norm(np.array(a)-np.array(b))

    def neighbors(self, p, occ):
        x,y = p
        neigh = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        return [n for n in neigh if 0 <= n[0] < occ.shape[1] 
                               and 0 <= n[1] < occ.shape[0]
                               and occ[n[1],n[0]] == 0]

    def plan(self, occ, start, goal):
        pq = []
        heapq.heappush(pq, (0,start))
        parent = {start: None}
        g = {start: 0}

        while pq:
            _, cur = heapq.heappop(pq)

            if cur == goal:
                break

            for n in self.neighbors(cur, occ):
                ng = g[cur] + 1
                if n not in g or ng < g[n]:
                    g[n] = ng
                    f = ng + self.h(n, goal)
                    heapq.heappush(pq, (f,n))
                    parent[n] = cur

        # reconstruct
        path = []
        cur = goal
        while cur:
            path.append(cur)
            cur = parent[cur]
        return path[::-1]
