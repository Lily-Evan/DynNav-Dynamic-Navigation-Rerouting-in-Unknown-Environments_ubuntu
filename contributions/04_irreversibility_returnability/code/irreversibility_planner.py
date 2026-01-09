# irreversibility_planner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq
import numpy as np

@dataclass
class PlannerResult:
    path: List[Tuple[int, int]]
    cost: float
    expansions: int
    success: bool
    reason: str

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def neighbors4(y: int, x: int, h: int, w: int):
    if y > 0: yield (y-1, x)
    if y < h-1: yield (y+1, x)
    if x > 0: yield (y, x-1)
    if x < w-1: yield (y, x+1)

def astar_irreversibility_constrained(
    free_mask: np.ndarray,
    irreversibility_grid: np.ndarray,
    start: Tuple[int,int],
    goal: Tuple[int,int],
    tau: float = 0.65,
    step_cost: float = 1.0,
    use_8conn: bool = False,
) -> PlannerResult:
    """
    A* with hard constraint:
      - only expand cells with I <= tau and free_mask True
    """
    free_mask = free_mask.astype(bool)
    I = np.asarray(irreversibility_grid, dtype=float)
    h, w = free_mask.shape

    sy, sx = start
    gy, gx = goal

    def in_bounds(p):
        y, x = p
        return 0 <= y < h and 0 <= x < w

    if not in_bounds(start) or not in_bounds(goal):
        return PlannerResult([], float("inf"), 0, False, "start/goal out of bounds")

    if (not free_mask[sy, sx]) or (not free_mask[gy, gx]):
        return PlannerResult([], float("inf"), 0, False, "start/goal is not free")

    if I[sy, sx] > tau:
        return PlannerResult([], float("inf"), 0, False, "start violates irreversibility threshold")

    if I[gy, gx] > tau:
        return PlannerResult([], float("inf"), 0, False, "goal violates irreversibility threshold")

    # priority queue: (f, g, (y,x))
    pq = []
    gscore: Dict[Tuple[int,int], float] = {start: 0.0}
    parent: Dict[Tuple[int,int], Tuple[int,int]] = {}
    heapq.heappush(pq, (manhattan(start, goal), 0.0, start))

    expansions = 0
    visited = set()

    while pq:
        f, g, cur = heapq.heappop(pq)
        if cur in visited:
            continue
        visited.add(cur)
        expansions += 1

        if cur == goal:
            # reconstruct
            path = [cur]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            path.reverse()
            return PlannerResult(path, g, expansions, True, "ok")

        cy, cx = cur
        for nb in neighbors4(cy, cx, h, w):
            ny, nx = nb

            if not free_mask[ny, nx]:
                continue
            if I[ny, nx] > tau:
                continue

            tentative_g = g + step_cost
            if tentative_g < gscore.get(nb, float("inf")):
                gscore[nb] = tentative_g
                parent[nb] = cur
                hval = manhattan(nb, goal)
                heapq.heappush(pq, (tentative_g + hval, tentative_g, nb))

    return PlannerResult([], float("inf"), expansions, False, "no path under irreversibility constraint")
