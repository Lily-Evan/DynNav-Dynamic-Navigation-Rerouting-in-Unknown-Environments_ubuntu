class DStarLite:
    """
    Very simplified D* Lite for dynamic replanning.
    """

    def __init__(self, grid):
        self.grid = grid

    def update_obstacle(self, x, y):
        self.grid[y][x] = 1

    def replan(self, start, goal):
        from modules.graph_planning.a_star import AStar
        astar = AStar(self.grid)
        return astar.plan(start, goal)
