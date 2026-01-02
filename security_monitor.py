# security_monitor.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Optional

import math
import numpy as np
from statistics import NormalDist


def _chi2_isf_wilson_hilferty(alpha: float, dof: int) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if dof <= 0:
        raise ValueError("dof must be positive")

    z = NormalDist().inv_cdf(1.0 - alpha)
    k = float(dof)
    a = 1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))
    return k * (a ** 3)


def mahalanobis_squared(nu: np.ndarray, S: np.ndarray, eps: float = 1e-9) -> float:
    nu = np.asarray(nu, dtype=float).reshape(-1)
    S = np.asarray(S, dtype=float)
    m = nu.shape[0]
    if S.shape != (m, m):
        raise ValueError(f"S must be shape ({m},{m}) but got {S.shape}")

    S_stable = S + eps * np.eye(m)
    try:
        x = np.linalg.solve(S_stable, nu)
        return float(nu.T @ x)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S_stable)
        return float(nu.T @ Sinv @ nu)


@dataclass
class MonitorConfig:
    alpha: float = 0.01
    window: int = 200
    warmup: int = 20
    min_dof: int = 1
    max_reasonable_d2: float = 1e6

    # Trigger policy:
    mode: str = "consecutive"   # "consecutive" or "kofn"

    # consecutive mode params
    consecutive: int = 3

    # kofn mode params
    k: int = 3
    n: int = 10


class InnovationMonitor:
    def __init__(self, cfg: Optional[MonitorConfig] = None):
        self.cfg = cfg or MonitorConfig()
        self._d2_hist: Deque[float] = deque(maxlen=self.cfg.window)
        self._updates = 0

        # consecutive tracking
        self._flag_streak = 0

        # K-of-N tracking (store last N flags)
        self._flag_window: Deque[int] = deque(maxlen=max(1, self.cfg.n))

        self._total = 0
        self._flags = 0

    def threshold(self, dof: int) -> float:
        dof = int(max(self.cfg.min_dof, dof))
        return _chi2_isf_wilson_hilferty(self.cfg.alpha, dof)

    def update(self, nu: np.ndarray, S: np.ndarray, *, dof: Optional[int] = None, meta: Optional[Dict] = None) -> Dict:
        nu = np.asarray(nu, dtype=float).reshape(-1)
        dof_val = int(dof) if dof is not None else int(nu.shape[0])
        dof_val = max(self.cfg.min_dof, dof_val)

        d2 = mahalanobis_squared(nu, S)
        if not np.isfinite(d2):
            d2 = self.cfg.max_reasonable_d2
        d2 = float(np.clip(d2, 0.0, self.cfg.max_reasonable_d2))

        thr = float(self.threshold(dof_val))
        flagged = bool(d2 > thr)

        self._total += 1
        if flagged:
            self._flags += 1

        self._updates += 1

        # update trackers
        if flagged:
            self._flag_streak += 1
        else:
            self._flag_streak = 0

        self._flag_window.append(1 if flagged else 0)

        # decide trigger
        triggered = False
        if self._updates > self.cfg.warmup:
            if self.cfg.mode == "consecutive":
                triggered = (self._flag_streak >= int(self.cfg.consecutive))
            elif self.cfg.mode == "kofn":
                k = int(self.cfg.k)
                # window size is up to N, but early on it's smaller
                triggered = (sum(self._flag_window) >= k)
            else:
                raise ValueError(f"Unknown mode: {self.cfg.mode}")

        self._d2_hist.append(d2)
        hist = np.array(self._d2_hist, dtype=float)
        rolling_mean = float(hist.mean()) if hist.size > 0 else 0.0
        rolling_std = float(hist.std(ddof=1)) if hist.size > 1 else 0.0

        return {
            "d2": d2,
            "thr": thr,
            "dof": dof_val,
            "flagged": flagged,
            "triggered": bool(triggered),
            "streak": int(self._flag_streak),
            "kofn_sum": int(sum(self._flag_window)),
            "flag_rate": float(self._flags / max(1, self._total)),
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "meta": meta or {},
        }
