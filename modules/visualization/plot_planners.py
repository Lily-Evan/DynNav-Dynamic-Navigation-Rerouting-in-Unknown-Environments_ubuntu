import numpy as np
import matplotlib.pyplot as plt
from modules.planning.a_star import AStar
from modules.planning.rrt import RRT
from modules.planning.rrt_star import RRTStar
import os

def plot_planners(save_path="data/plots/planners.png"):
    h, w = 60, 80
    occ = np.zeros((h, w), dtype=int)
    occ[20:40, 30] = 1
    occ[10:50, 55] = 1

    start = (5, 5)
    goal  = (70, 50)

    a_star = AStar()
    path_a = a_star.plan(occ, start, goal)

    rrt = RRT(step=3, max_iter=2000)
    path_rrt = rrt.plan(occ, start, goal)

    rrt_star = RRTStar(step=3, radius=10, max_iter=2000)
    path_rrt_star = rrt_star.plan(occ, start, goal)

    plt.figure(figsize=(7,5))
    plt.imshow(occ, cmap='gray_r')

    if path_a:
        pa = np.array(path_a)
        plt.plot(pa[:,0], pa[:,1], label="A*", linewidth=2)
    if path_rrt:
        pr = np.array(path_rrt)
        plt.plot(pr[:,0], pr[:,1], label="RRT", linewidth=2)
    if path_rrt_star:
        prs = np.array(path_rrt_star)
        plt.plot(prs[:,0], prs[:,1], label="RRT*", linewidth=2)

    plt.scatter([start[0]],[start[1]], c='green', s=50, label="Start")
    plt.scatter([goal[0]],[goal[1]], c='red', s=50, label="Goal")

    plt.legend()
    plt.title("Planner Paths Comparison")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

if __name__ == "__main__":
    print("Saved:", plot_planners())
