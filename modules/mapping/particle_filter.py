import numpy as np

class ParticleFilter:
    def __init__(self, n=300):
        self.n = n
        self.particles = np.random.uniform(0, 10, (n, 3))
        self.weights = np.ones(n) / n

    def predict(self, u):
        self.particles[:,0] += u[0] + np.random.randn(self.n)*0.02
        self.particles[:,1] += u[1] + np.random.randn(self.n)*0.02

    def update(self, z, landmark):
        dist = np.linalg.norm(self.particles[:,:2] - landmark, axis=1)
        error = np.abs(dist - z)

        self.weights = np.exp(-error)
        self.weights /= np.sum(self.weights)

    def resample(self):
        idx = np.random.choice(self.n, self.n, p=self.weights)
        self.particles = self.particles[idx]
        self.weights = np.ones(self.n)/self.n

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)
