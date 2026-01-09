# ================================================================
# Multi-Robot Shared Uncertainty Model (Conceptual Demo)
# ================================================================

from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
class Robot:
    name: str
    drift_estimate: float
    uncertainty_estimate: float

    def get_state(self):
        return self.drift_estimate, self.uncertainty_estimate


def fuse_uncertainty(robots: List[Robot]):
    """
    Απλό fusion:
    - weighted average drift
    - uncertainty fusion ως harmonic mean (πιο conservative)
    """

    drifts = np.array([r.drift_estimate for r in robots])
    vars_ = np.array([r.uncertainty_estimate for r in robots])

    fused_drift = np.mean(drifts)

    # harmonic → μικρό variance μετράει περισσότερο
    fused_var = len(vars_) / np.sum(1.0 / vars_)

    return fused_drift, fused_var
