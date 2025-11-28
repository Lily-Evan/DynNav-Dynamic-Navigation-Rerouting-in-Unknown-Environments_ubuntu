import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.memory = deque(maxlen=50000)
        self.gamma = 0.98
        self.eps = 1.0
        self.eps_min = 0.05

        # Simple linear model
        self.W = np.random.randn(state_dim, action_dim) * 0.1

    def act(self, state):
        if random.random() < self.eps:
            return random.randrange(self.action_dim)
        return np.argmax(state @ self.W)

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state in batch:
            target = reward + self.gamma * np.max(next_state @ self.W)
            pred = state @ self.W
            error = target - pred[action]

            # gradient update
            self.W[:,action] += 0.001 * error * state

        if self.eps > self.eps_min:
            self.eps *= 0.995

