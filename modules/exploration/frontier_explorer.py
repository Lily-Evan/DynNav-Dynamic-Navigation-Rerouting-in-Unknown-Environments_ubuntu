import numpy as np
from scipy.ndimage import label

class FrontierExplorer:
    def __init__(self, occ_grid):
        """
        occ_grid: Occupancy grid
        -1 = unknown
        0 = free
        100 = obstacle
        """
        self.grid = occ_grid

    def get_frontiers(self):
        unknown = (self.grid == -1)
        free = (self.grid == 0)

        frontier = np.zeros_like(self.grid)

        for y in range(1, self.grid.shape[0]-1):
            for x in range(1, self.grid.shape[1]-1):
                if free[y, x]:
                    if np.any(unknown[y-1:y+2, x-1:x+2]):
                        frontier[y, x] = 1

        clusters, n = label(frontier)
        frontier_list = []

        for i in range(1, n + 1):
            pts = np.argwhere(clusters == i)
            frontier_list.append(pts)

        return frontier_list
