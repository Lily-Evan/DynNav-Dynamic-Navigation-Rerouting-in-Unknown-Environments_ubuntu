import numpy as np

class PriorityField:
    def __init__(self, coverage_grid, uncertainty_grid, w_cov=0.7, w_unc=0.3):
        self.cov = coverage_grid
        self.unc = uncertainty_grid
        self.w_cov = w_cov
        self.w_unc = w_unc

    def compute(self):
        return self.w_cov*(1 - self.cov) + self.w_unc*self.unc
