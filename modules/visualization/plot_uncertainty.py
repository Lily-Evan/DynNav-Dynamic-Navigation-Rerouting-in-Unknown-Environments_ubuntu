import numpy as np
import matplotlib.pyplot as plt
import os

def plot_uncertainty(save_path="data/plots/uncertainty.png"):
    h, w = 50, 50
    yy, xx = np.mgrid[0:h, 0:w]
    unc = np.exp(-((xx-35)**2 + (yy-15)**2)/(2*8**2))
    unc = (unc - unc.min())/(unc.max()-unc.min()+1e-6)

    plt.figure(figsize=(6,5))
    im = plt.imshow(unc, cmap="viridis")
    plt.colorbar(im, label="Uncertainty")
    plt.title("Uncertainty Map")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

if __name__ == "__main__":
    print("Saved:", plot_uncertainty())
