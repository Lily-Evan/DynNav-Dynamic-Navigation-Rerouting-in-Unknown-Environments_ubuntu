import torch
import torch.nn as nn
import torch.nn.functional as F

class HeuristicNet(nn.Module):
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # cost-to-go
