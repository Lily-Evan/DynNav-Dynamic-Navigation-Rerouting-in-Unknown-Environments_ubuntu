import matplotlib.pyplot as plt
import numpy as np
from modules.utils.nce_monitor import NCEMonitor
import os

def plot_nce(save_path="data/plots/nce.png"):
    mon = NCEMonitor()
    for t in range(200):
        cov = 1 - np.exp(-t/60)
        path = 1 + t*0.02
        mon.update(cov, path)

    cov, path, nce = mon.get_arrays()

    plt.figure(figsize=(6,4))
    plt.plot(nce)
    plt.xlabel("Time (steps)")
    plt.ylabel("NCE")
    plt.title("NCE Over Time")
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

if __name__ == "__main__":
    out = plot_nce()
    print("Saved:", out)
