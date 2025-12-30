# ================================================================
# Multi-Robot Cooperative Planning Demo
# ================================================================

from multi_robot_cooperative_planning import (
    Robot,
    RobotPathEstimate,
    cooperative_choose_path,
)
from risk_explainer import PathStats, explain_preference


def main():
    # Ορίζουμε nominal paths (μόνο για length εδώ)
    base_paths = [
        PathStats("A", length=10.0, drift_exposure=0.0, uncertainty_exposure=0.0),
        PathStats("B", length=11.0, drift_exposure=0.0, uncertainty_exposure=0.0),
        PathStats("C", length=13.0, drift_exposure=0.0, uncertainty_exposure=0.0),
    ]

    # Robot 1: "αισιόδοξο" για path A, πιο επιφυλακτικό για C
    r1_estimates = {
        "A": RobotPathEstimate("R1", "A", drift_exposure=4.0, uncertainty_exposure=0.001),
        "B": RobotPathEstimate("R1", "B", drift_exposure=3.0, uncertainty_exposure=0.002),
        "C": RobotPathEstimate("R1", "C", drift_exposure=2.5, uncertainty_exposure=0.003),
    }

    # Robot 2: εμπιστεύεται περισσότερο C, θεωρεί A πιο risk
    r2_estimates = {
        "A": RobotPathEstimate("R2", "A", drift_exposure=6.0, uncertainty_exposure=0.003),
        "B": RobotPathEstimate("R2", "B", drift_exposure=3.5, uncertainty_exposure=0.0025),
        "C": RobotPathEstimate("R2", "C", drift_exposure=1.8, uncertainty_exposure=0.0015),
    }

    robot1 = Robot(name="R1", path_estimates=r1_estimates)
    robot2 = Robot(name="R2", path_estimates=r2_estimates)

    robots = [robot1, robot2]

    lambda_risk = 1.0

    print("\n================ ROBOT PATH ESTIMATES ================")
    for r in robots:
        print(f"\n{r.name}:")
        for pname, est in r.path_estimates.items():
            print(
                f"  Path {pname}: drift={est.drift_exposure:.3f}, "
                f"var={est.uncertainty_exposure:.6f}"
            )

    # Cooperative decision
    best_fused, fused_paths = cooperative_choose_path(
        robots=robots,
        base_paths=base_paths,
        lambda_risk=lambda_risk,
    )

    print("\n================ FUSED PATH ESTIMATES ================")
    for p in fused_paths:
        print(
            f"Path {p.name}: length={p.length:.2f}, "
            f"fused drift={p.drift_exposure:.3f}, "
            f"fused var={p.uncertainty_exposure:.6f}"
        )

    print("\n================ COOPERATIVE DECISION ================")
    print(f"Selected path (cooperative) = {best_fused.name}")

    print("\n================ EXPLANATION ==========================")
    for p in fused_paths:
        if p.name != best_fused.name:
            print(f"\n{best_fused.name} vs {p.name}:")
            print(" ", explain_preference(best_fused, p, lambda_risk=lambda_risk))

    print("\nΣυμπέρασμα:")
    print("Η απόφαση προκύπτει από συγχώνευση εκτιμήσεων drift/uncertainty")
    print("δύο ρομπότ και risk-aware κριτήριο κόστους με explainable output.\n")


if __name__ == "__main__":
    main()
