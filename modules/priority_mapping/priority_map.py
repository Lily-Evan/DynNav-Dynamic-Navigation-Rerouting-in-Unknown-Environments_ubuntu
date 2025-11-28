#!/usr/bin/env python3
import numpy as np

class PriorityField:
    def __init__(self, w_uncov=0.6, w_uncert=0.4):
        self.w_uncov = w_uncov
        self.w_uncert = w_uncert

    def compute_priority(self, coverage_grid, uncertainty_grid):
        """
        coverage_grid: binary 0=uncovered, 1=covered
        uncertainty_grid: 0..1
        """

        uncovered = 1 - coverage_grid
        priority = self.w_uncov*uncovered + self.w_uncert*uncertainty_grid

        # Normalize to 0..1
        priority = (priority - priority.min()) / (priority.max() - priority.min() + 1e-6)

        return priority
