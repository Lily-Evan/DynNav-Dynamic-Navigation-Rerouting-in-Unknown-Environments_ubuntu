import numpy as np
import matplotlib.pyplot as plt
from modules.exploration.nbv import NextBestView
import os

def plot_nbv(save_path="data/plots/nbv.png"):
    h, w = 50, 50
    coverage = np.random.rand(h,w)
    uncertainty = np.random.rand(h,w)

    nbv = NextBestView(coverage, uncertainty)
    S = nbv.score()
    iy, ix = nbv.select_nbv()

    plt.figure(figsize=(6,6))
    im = plt.imshow(S, cmap="plasma")
    plt.colorbar(im, label="NBV score")
    plt.scatter(ix, iy, s=60, edgecolors='black', facecolors='white', label="NBV")
    plt.legend(loc="upper right", fontsize=8)
    plt.title("Next-Best-View Heatmap")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

if __name__ == "__main__":
    out = plot_nbv()
    print("Saved:", out)
