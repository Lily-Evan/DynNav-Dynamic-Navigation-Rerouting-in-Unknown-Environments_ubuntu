import numpy as np
import matplotlib.pyplot as plt
from modules.exploration.frontier_explorer import FrontierExplorer
import os

def plot_frontiers(save_path="data/plots/frontiers.png"):
    grid = np.zeros((100,100))
    grid[:,40:45] = -1      # unknown stripe
    grid[20:80, 20] = 100   # obstacle wall

    F = FrontierExplorer(grid)
    frontiers = F.get_frontiers()

    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap='gray')
    for f in frontiers:
        pts = np.array(f)
        plt.scatter(pts[:,1], pts[:,0], s=4, color='red')

    plt.title("Frontier Cells")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

if __name__ == "__main__":
    out = plot_frontiers()
    print("Saved:", out)
