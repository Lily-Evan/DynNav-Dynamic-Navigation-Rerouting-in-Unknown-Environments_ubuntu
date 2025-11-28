import numpy as np

class ICP:
    def run(self, A, B, iterations=20):
        """
        A: source points (Nx2)
        B: target points (Nx2)
        """
        T = np.eye(3)

        for _ in range(iterations):
            # manual nearest neighbors (χωρίς sklearn)
            B_matched = []
            for a in A:
                dists = np.linalg.norm(B - a, axis=1)
                j = np.argmin(dists)
                B_matched.append(B[j])
            B_matched = np.array(B_matched)

            A_mean = A.mean(axis=0)
            B_mean = B_matched.mean(axis=0)

            H = (A - A_mean).T @ (B_matched - B_mean)
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            t = B_mean - R @ A_mean

            step = np.eye(3)
            step[:2,:2] = R
            step[:2, 2] = t
            T = step @ T

            # apply transform
            A = (R @ A.T).T + t

        return T
