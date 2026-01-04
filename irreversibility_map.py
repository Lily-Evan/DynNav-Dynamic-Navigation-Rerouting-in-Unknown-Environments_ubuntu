# irreversibility_map.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class IrreversibilityConfig:
    """
    Irreversibility score I(row,col) in [0,1]
    Higher => entering that cell is more likely to be "point-of-no-return".
    """
    w_uncert: float = 0.60      # uncertainty / drift proxy
    w_sparsity: float = 0.25    # low feature density proxy (optional)
    w_deadend: float = 0.15     # topology / dead-end propensity
    eps: float = 1e-9
    deadend_radius: int = 2


def _normalize01(a: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    amin = float(np.min(a))
    amax = float(np.max(a))
    if (amax - amin) < eps:
        return np.zeros_like(a)
    return (a - amin) / (amax - amin)


def estimate_deadend_score(free_mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Simple topology proxy:
    Cells with fewer free neighbors in a local window get higher score.
    """
    free = free_mask.astype(np.uint8)
    h, w = free.shape
    score = np.zeros((h, w), dtype=float)

    r = max(1, int(radius))
    for y in range(h):
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        for x in range(w):
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            window = free[y0:y1, x0:x1]
            free_count = int(window.sum())
            # fewer "escape options" => higher dead-end score
            score[y, x] = 1.0 / (free_count + 1.0)

    return _normalize01(score)


def load_grid_from_cell_table_csv(
    path: str,
    value_col: str = "uncertainty",
    row_col: tuple[str, str] = ("row", "col"),
    fill_nan_with: str = "max",
) -> np.ndarray:
    """
    Loads a CSV like:
      cell_id,row,col,center_x,center_z,covered,uncertainty
      0,0,0,...,0,0.8
      ...

    Returns grid[row, col] = value_col.

    fill_nan_with:
      - "max": fill missing cells with max(grid)
      - "zero": fill missing cells with 0
    """
    import pandas as pd

    df = pd.read_csv(path)
    rname, cname = row_col

    if rname not in df.columns or cname not in df.columns:
        raise ValueError(f"CSV missing required columns {row_col}. Found: {list(df.columns)}")
    if value_col not in df.columns:
        raise ValueError(f"CSV missing value_col='{value_col}'. Found: {list(df.columns)}")

    rows = df[rname].astype(int).to_numpy()
    cols = df[cname].astype(int).to_numpy()
    vals = df[value_col].astype(float).to_numpy()

    h = int(rows.max()) + 1
    w = int(cols.max()) + 1
    grid = np.full((h, w), np.nan, dtype=float)
    grid[rows, cols] = vals

    if np.isnan(grid).any():
        if fill_nan_with == "max":
            mx = np.nanmax(grid)
            grid = np.nan_to_num(grid, nan=mx)
        elif fill_nan_with == "zero":
            grid = np.nan_to_num(grid, nan=0.0)
        else:
            raise ValueError("fill_nan_with must be 'max' or 'zero'")

    return grid


def load_covered_mask_from_cell_table_csv(
    path: str,
    covered_col: str = "covered",
    row_col: tuple[str, str] = ("row", "col"),
) -> np.ndarray:
    """
    Returns covered_mask[row,col] as bool from a 'covered' column (0/1).
    """
    import pandas as pd

    df = pd.read_csv(path)
    rname, cname = row_col

    if rname not in df.columns or cname not in df.columns:
        raise ValueError(f"CSV missing required columns {row_col}. Found: {list(df.columns)}")
    if covered_col not in df.columns:
        raise ValueError(f"CSV missing covered_col='{covered_col}'. Found: {list(df.columns)}")

    rows = df[rname].astype(int).to_numpy()
    cols = df[cname].astype(int).to_numpy()
    cov = df[covered_col].astype(int).to_numpy()

    h = int(rows.max()) + 1
    w = int(cols.max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[rows, cols] = (cov > 0)
    return mask


def build_irreversibility_map(
    uncertainty_grid: np.ndarray,
    free_mask: np.ndarray,
    feature_density_grid: np.ndarray | None = None,
    cfg: IrreversibilityConfig = IrreversibilityConfig(),
) -> np.ndarray:
    """
    I = w_uncert * norm(uncertainty) + w_sparsity * (1-norm(feature_density)) + w_deadend * deadend_score
    """
    unc01 = _normalize01(uncertainty_grid, cfg.eps)

    if feature_density_grid is None:
        sparsity01 = np.zeros_like(unc01)
    else:
        fd01 = _normalize01(feature_density_grid, cfg.eps)
        sparsity01 = 1.0 - fd01

    dead01 = estimate_deadend_score(free_mask=free_mask, radius=cfg.deadend_radius)

    I = (cfg.w_uncert * unc01) + (cfg.w_sparsity * sparsity01) + (cfg.w_deadend * dead01)
    I = np.clip(I, 0.0, 1.0)

    # Not-free cells are "forbidden / catastrophic"
    I = np.where(free_mask, I, 1.0)
    return I

