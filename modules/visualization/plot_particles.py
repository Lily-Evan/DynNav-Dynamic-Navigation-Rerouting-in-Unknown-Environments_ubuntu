import numpy as np
import matplotlib.pyplot as plt
from modules.mapping.particle_filter import ParticleFilter
import os

def plot_particles(save_path="data/plots/particles.png"):
    pf = ParticleFilter(n=300)
    landmark = np.array([5.0, 5.0])

    for _ in range(10):
        pf.predict([0.2, 0.1])
        pf.update(z=5.0, landmark=landmark)
        pf.resample()

    est = pf.estimate()

    plt.figure(figsize=(6,6))
    plt.scatter(pf.particles[:,0], pf.particles[:,1], s=5, alpha=0.5, label="Particles")
    plt.scatter([landmark[0]], [landmark[1]], c='red', s=60, label="Landmark")
    plt.scatter([est[0]], [est[1]], c='green', s=60, label="Estimate")
    plt.legend()
    plt.title("Particle Filter State")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

if __name__ == "__main__":
    print("Saved:", plot_particles())
