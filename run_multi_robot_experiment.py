# ================================================================
# Multi Robot Uncertainty Cooperation Experiment
# ================================================================

from multi_robot_uncertainty import Robot, fuse_uncertainty


def main():
    # -------- ROBOT STATES ----------
    robot1 = Robot(
        name="R1",
        drift_estimate=0.8,
        uncertainty_estimate=0.002,
    )

    robot2 = Robot(
        name="R2",
        drift_estimate=0.3,
        uncertainty_estimate=0.006,
    )

    robots = [robot1, robot2]

    print("\n================ ROBOT STATES ================")
    for r in robots:
        d, u = r.get_state()
        print(f"{r.name}: drift={d:.3f}, var={u:.6f}")

    # --------- FUSION ----------
    fused_drift, fused_var = fuse_uncertainty(robots)

    print("\n================ FUSION RESULT ================")
    print(f"Fused Drift  : {fused_drift:.4f}")
    print(f"Fused VAR    : {fused_var:.6f}")

    # --------- INTERPRETATION ----------
    print("\n================ INTERPRETATION ================")

    if fused_var < min(r.uncertainty_estimate for r in robots):
        print("⚠️  Το fused variance είναι μικρότερο από κάθε robot → επικίνδυνο (overconfidence).")
    else:
        print("✔️ Το fused variance παραμένει ρεαλιστικό.")

    if fused_drift < min(r.drift_estimate for r in robots):
        print("⚠️  Το σύστημα έγινε υπερ-αισιόδοξο ως προς drift.")
    else:
        print("✔️ Το fused drift είναι ρεαλιστικό.")

    print("\nΤο συμπέρασμα:")
    print("Κοινή γνώση = πιο σταθερή εκτίμηση drift + καλύτερη συνολική αβεβαιότητα.")
    print("Αυτό ανοίγει δρόμο για cooperative planning και shared risk management.\n")


if __name__ == "__main__":
    main()
