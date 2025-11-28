import numpy as np

class NCEMonitor:
    def __init__(self, threshold=0.001):
        self.coverage = []
        self.path_length = []
        self.NCE = []
        self.threshold = threshold

    def update(self, cov, path):
        self.coverage.append(cov)
        self.path_length.append(path)
        nce = cov / (path + 1e-9)
        self.NCE.append(nce)

        return nce < self.threshold

    def get_arrays(self):
        return (
            np.array(self.coverage),
            np.array(self.path_length),
            np.array(self.NCE)
        )
