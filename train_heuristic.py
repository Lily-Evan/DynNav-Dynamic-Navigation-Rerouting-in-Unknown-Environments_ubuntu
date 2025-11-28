import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from learned_heuristic import HeuristicNet

class PlannerDataset(Dataset):
    def __init__(self, npz_path="planner_dataset.npz"):
        data = np.load(npz_path)
        self.X = data["X"]
        self.y = data["y"]

        # normalize λίγο τα inputs (προαιρετικό)
        self.X_mean = self.X.mean(axis=0)
        self.X_std = self.X.std(axis=0) + 1e-6
        self.X = (self.X - self.X_mean) / self.X_std

        np.savez("planner_dataset_norm_stats.npz",
                 mean=self.X_mean, std=self.X_std)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.float32)
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y

def train():
    ds = PlannerDataset("planner_dataset.npz")
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    net = HeuristicNet(in_dim=4, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(30):
        total_loss = 0.0
        for Xb, yb in dl:
            pred = net(Xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(Xb)
        print(f"Epoch {epoch+1}: loss = {total_loss / len(ds):.4f}")

    torch.save(net.state_dict(), "heuristic_net.pt")
    print("Saved heuristic_net.pt")

if __name__ == "__main__":
    train()
