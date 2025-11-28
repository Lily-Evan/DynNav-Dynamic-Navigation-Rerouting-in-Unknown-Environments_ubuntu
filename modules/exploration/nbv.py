import numpy as np

class NextBestView:
    def __init__(self, coverage, uncertainty):
        self.coverage = coverage
        self.uncertainty = uncertainty

    def score(self):
        # όσο λιγότερη κάλυψη + όσο περισσότερη αβεβαιότητα → τόσο καλύτερο NBV
        return 0.7 * (1.0 - self.coverage) + 0.3 * self.uncertainty

    def select_nbv(self):
        S = self.score()
        idx = np.unravel_index(np.argmax(S), S.shape)
        return idx
