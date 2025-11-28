import numpy as np

class PurePursuit:
    def __init__(self, lookahead=0.5):
        self.lookahead = lookahead

    def compute(self, pose, path):
        x, y, yaw = pose

        if len(path) == 0:
            return 0.0

        dists = [np.linalg.norm(np.array([px, py]) - np.array([x, y])) for px, py in path]
        mask = np.array(dists) > self.lookahead

        if not np.any(mask):
            idx = np.argmax(dists)
        else:
            idx = np.argmax(np.where(mask, dists, -1))

        target = path[idx]

        dx = target[0] - x
        dy = target[1] - y

        angle = np.arctan2(dy, dx)
        steering = angle - yaw
        return steering
