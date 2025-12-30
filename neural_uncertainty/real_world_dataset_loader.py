# ================================================================
# Real World Dataset Loader
# Υποστηρίζει raw CSV logs -> standardized drift dataset
# ================================================================

import os
import pandas as pd
import numpy as np


class RealWorldDatasetLoader:

    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        print(f"[RW-INFO] Loaded dataset: {self.df.shape[0]} samples, {len(self.df.columns)} columns")

    def normalize_column(self, col):
        values = self.df[col].values.astype(np.float32)
        m = np.mean(values)
        s = np.std(values) + 1e-8
        return (values - m) / s

    def build_feature_matrix(
        self,
        feature_cols,
        target_col
    ):
        if target_col not in self.df.columns:
            raise ValueError(f"Target column {target_col} not found")

        for c in feature_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing feature column: {c}")

        X = np.stack([self.normalize_column(c) for c in feature_cols], axis=1)
        y = self.df[target_col].values.astype(np.float32)

        print(f"[RW-INFO] Built dataset: X={X.shape}, y={y.shape}")
        return X, y
