import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


CSV_PATH = "drift_dataset.csv"
MODEL_PATH = "drift_uncertainty_net.pt"
NORM_STATS_PATH = "drift_uncertainty_norm_stats.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DriftDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        # ----- Επιλογή target column -----
        # Προσπαθούμε αυτόματα:
        if "drift_norm" in df.columns:
            target_col = "drift_norm"
        elif "drift" in df.columns:
            target_col = "drift"
        else:
            raise ValueError(
                "Δεν βρήκα στήλη 'drift_norm' ή 'drift' στο drift_dataset.csv.\n"
                "Άνοιξε το CSV και άλλαξε χειροκίνητα το κομμάτι επιλογής target_col."
            )

        self.target_col = target_col

        # Features = όλα τα numeric columns εκτός target
        num_df = df.select_dtypes(include=[np.number])
        feature_cols = [c for c in num_df.columns if c != target_col]

        if len(feature_cols) == 0:
            raise ValueError("Δεν βρήκα numeric feature columns στο drift_dataset.csv.")

        self.feature_cols = feature_cols

        X = num_df[feature_cols].to_numpy(dtype=np.float32)
        y = num_df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

        # Αποθήκευση normalization stats
        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std = X.std(axis=0, keepdims=True) + 1e-8

        self.y_mean = y.mean(axis=0, keepdims=True)
        self.y_std = y.std(axis=0, keepdims=True) + 1e-8

        Xn = (X - self.x_mean) / self.x_std
        yn = (y - self.y_mean) / self.y_std

        self.X = torch.from_numpy(Xn)
        self.y = torch.from_numpy(yn)

        print("[INFO] Using target column:", self.target_col)
        print("[INFO] Feature columns:", self.feature_cols)
        print("[INFO] Dataset size:", self.X.shape[0])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class UncertaintyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers)

        # output: mean and log_std (1-dim target)
        self.head = nn.Linear(prev, 2)

    def forward(self, x):
        h = self.backbone(x)
        out = self.head(h)
        mean = out[:, :1]
        log_std = out[:, 1:]
        # clamp log_std to avoid crazy values
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mean, log_std


def gaussian_nll(y, mean, log_std):
    # y, mean, log_std: [B, 1]
    var = torch.exp(2 * log_std)
    return 0.5 * (math.log(2 * math.pi) + 2 * log_std + (y - mean) ** 2 / var)


def train():
    full_dataset = DriftDataset(CSV_PATH)

    # Train/val split (80/20)
    n = len(full_dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = UncertaintyMLP(input_dim=len(full_dataset.feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")

    for epoch in range(1, 51):
        model.train()
        train_loss = 0.0
        n_train_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            mean, log_std = model(xb)
            loss = gaussian_nll(yb, mean, log_std).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            n_train_samples += xb.size(0)

        train_loss /= n_train_samples

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                mean, log_std = model(xb)
                loss = gaussian_nll(yb, mean, log_std).mean()
                val_loss += loss.item() * xb.size(0)
                n_val_samples += xb.size(0)
        val_loss /= n_val_samples

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model + normalization stats
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": len(full_dataset.feature_cols),
                    "feature_cols": full_dataset.feature_cols,
                    "target_col": full_dataset.target_col,
                },
                MODEL_PATH,
            )
            np.savez(
                NORM_STATS_PATH,
                x_mean=full_dataset.x_mean,
                x_std=full_dataset.x_std,
                y_mean=full_dataset.y_mean,
                y_std=full_dataset.y_std,
            )
            print(f"[INFO] Saved best model to {MODEL_PATH}, stats to {NORM_STATS_PATH}")

    print("[INFO] Training finished. Best val loss:", best_val_loss)


if __name__ == "__main__":
    train()
