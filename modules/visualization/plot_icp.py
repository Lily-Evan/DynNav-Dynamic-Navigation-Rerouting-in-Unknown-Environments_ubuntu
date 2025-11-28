import numpy as np
import matplotlib.pyplot as plt
from modules.mapping.icp import ICP
import os

def plot_icp(save_path="data/plots/icp.png"):
    t = np.linspace(0, 2*np.pi, 100)
    B = np.vstack([np.cos(t), np.sin(t)]).T

    R = np.array([[np.cos(0.5), -np.sin(0.5)],
                  [np.sin(0.5),  np.cos(0.5)]])
    A = B @ R.T + np.array([0.5, 0.2])

    icp = ICP()
    T = icp.run(A.copy(), B.copy(), iterations=20)

    A_h = np.hstack([A, np.ones((A.shape[0],1))])
    A_aligned = (T @ A_h.T).T[:,:2]

    plt.figure(figsize=(6,6))
    plt.scatter(B[:,0], B[:,1], c='blue', label="Target")
    plt.scatter(A[:,0], A[:,1], c='red', alpha=0.4, label="Source (before)")
    plt.scatter(A_aligned[:,0], A_aligned[:,1], c='green', alpha=0.7, label="Source (after)")
    plt.legend()
    plt.title("ICP Alignment")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

if __name__ == "__main__":
    print("Saved:", plot_icp())
