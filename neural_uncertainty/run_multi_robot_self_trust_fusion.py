# ================================================================
# Multi-Robot Fusion Weighted by Self-Trust
# ================================================================

from multi_robot_cooperative_planning import Robot, RobotPathEstimate, cooperative_choose_path
from risk_explainer import PathStats


def main():
    # nominal paths
    base_paths = [
        PathStats("A", length=10.0, drift_exposure=0.0, uncertainty_exposure=0.0),
        PathStats("B", length=11.0, drift_exposure=0.0, uncertainty_exposure=0.0),
        PathStats("C", length=13.0, drift_exposure=0.0, uncertainty_exposure=0.0),
    ]

    # Robot estimates as before
    r1_estimates = {
        "A": RobotPathEstimate("R1", "A", drift_exposure=4.0, uncertainty_exposure=0.001),
        "B": RobotPathEstimate("R1", "B", drift_exposure=3.0, uncertainty_exposure=0.002),
        "C": RobotPathEstimate("R1", "C", drift_exposure=2.5, uncertainty_exposure=0.003),
    }
    r2_estimates = {
        "A": RobotPathEstimate("R2", "A", drift_exposure=6.0, uncertainty_exposure=0.003),
        "B": RobotPathEstimate("R2", "B", drift_exposure=3.5, uncertainty_exposure=0.0025),
        "C": RobotPathEstimate("R2", "C", drift_exposure=1.8, uncertainty_exposure=0.0015),
    }

    # Self-Trust per robot (π.χ. R1 πιο αξιόπιστο)
    S1 = 0.8
    S2 = 0.4

    print("\n[MR-ST] Robot Self-Trust:")
    print(f"  R1: S={S1}")
    print(f"  R2: S={S2}")

    # Weighted fusion "by hand" για path A (ενδεικτικό)
    def weighted(d1, d2, S1, S2):
        w1 = S1 / (S1 + S2)
        w2 = S2 / (S1 + S2)
        return w1 * d1 + w2 * d2

    dA_fused = weighted(r1_estimates["A"].drift_exposure, r2_estimates["A"].drift_exposure, S1, S2)
    print(f"\n[MR-ST] Weighted fused drift for path A = {dA_fused:.3f}")

    print("\nΙδέα: Μπορείς να επεκτείνεις το cooperative_choose_path ώστε")
    print("να βάζει weights ανά robot στο fuse_path_estimates, ανάλογα με Sᵢ.")


if __name__ == "__main__":
    main()
