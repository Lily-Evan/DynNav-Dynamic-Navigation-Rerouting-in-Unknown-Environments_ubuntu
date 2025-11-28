#!/usr/bin/env python3
import numpy as np

class VOUncertainty:
    def __init__(self):
        self.history_inliers = []
        self.history_error = []

    def compute(self, n_inliers, repro_error):
        # Normalize scores
        inlier_score = np.tanh(n_inliers / 50)
        repro_score = np.tanh(1.0 / (repro_error + 1e-6))

        # Combined uncertainty
        uncertainty = 1.0 - (0.6*inlier_score + 0.4*repro_score)
        uncertainty = np.clip(uncertainty, 0, 1)

        self.history_inliers.append(n_inliers)
        self.history_error.append(repro_error)

        return uncertainty
