import numpy as np
from ukf_fusion import UKF


def run_sim(seed=0, T=200, attack_start=100, bias=None):
    np.random.seed(seed)
    ukf = UKF()

    x_true = np.array([0.0, 0.0, 0.0], dtype=float)

    dt = 1.0
    u = np.array([0.05, 0.02, 0.01], dtype=float)

    detected_at = None
    triggered_steps = 0
    flagged_steps = 0

    for t in range(T):
        x_true = x_true + np.array([0.05, 0.02, 0.01])
        ukf.predict(u, dt)

        z_vo = x_true + np.random.multivariate_normal(np.zeros(3), ukf.R_vo_nominal)

        if bias is not None and t >= attack_start:
            z_vo = z_vo + bias

        yaw_meas = float(x_true[2] + np.random.normal(0.0, np.sqrt(ukf.R_imu_nominal[0, 0])))

        ukf.update_vo(z_vo)
        ukf.update_imu(yaw_meas)

        info = ukf.last_vo_ids
        if info["flagged"]:
            flagged_steps += 1
        if info["triggered"]:
            triggered_steps += 1

        if ukf.security_alert and detected_at is None:
            detected_at = t

    # delay is meaningful only if there is an attack
    delay = None
    if bias is not None and detected_at is not None:
        delay = detected_at - attack_start

    return {
        "detected_at": detected_at,
        "delay": delay,
        "flagged_steps": flagged_steps,
        "triggered_steps": triggered_steps,
        "flag_rate": flagged_steps / T,
        "trigger_rate": triggered_steps / T,
    }


def main():
    T = 200
    attack_start = 100

    cases = [
        ("no_attack", None),
        ("small_bias", np.array([0.3, 0.3, 0.1])),
        ("med_bias", np.array([0.6, 0.6, 0.2])),
        ("big_bias", np.array([1.0, 1.0, 0.5])),
    ]

    print(f"T={T} attack_start={attack_start}")
    print("case, detected_at, delay, flag_rate, trigger_rate")

    for name, bias in cases:
        out = run_sim(seed=0, T=T, attack_start=attack_start, bias=bias)
        print(f"{name}, {out['detected_at']}, {out['delay']}, {out['flag_rate']:.3f}, {out['trigger_rate']:.3f}")


if __name__ == "__main__":
    main()
