import numpy as np
from collections import deque
from ukf_fusion import UKF


def run_replay(seed=0, T=200, attack_start=100, replay_len=30):
    np.random.seed(seed)
    ukf = UKF()

    x_true = np.array([0.0, 0.0, 0.0], dtype=float)

    dt = 1.0
    u = np.array([0.05, 0.02, 0.01], dtype=float)

    buf = deque(maxlen=replay_len)

    detected_at = None
    flagged_steps = 0
    triggered_steps = 0

    for t in range(T):
        x_true = x_true + np.array([0.05, 0.02, 0.01])
        ukf.predict(u, dt)

        z_fresh = x_true + np.random.multivariate_normal(np.zeros(3), ukf.R_vo_nominal)

        if t < attack_start:
            buf.append(z_fresh.copy())
            z_vo = z_fresh
        else:
            if len(buf) == 0:
                z_vo = z_fresh
            else:
                idx = (t - attack_start) % len(buf)
                z_vo = list(buf)[idx]

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

        if (t % 10 == 0) or (attack_start - 3 <= t <= attack_start + 8):
            ratio = info["d2"] / max(1e-9, info["thr"])
            print(
                f"t={t:03d} replay={'YES' if t>=attack_start else 'no '} "
                f"d2={info['d2']:.3f} thr={info['thr']:.3f} "
                f"ratio={ratio:.2f} scale={ukf.vo_scale_last:.2f} "
                f"flagged={info['flagged']} streak={info['streak']} trig={info['triggered']} "
                f"alert={ukf.security_alert}"
            )

    delay = None
    if detected_at is not None:
        delay = detected_at - attack_start

    return {
        "detected_at": detected_at,
        "delay": delay,
        "flag_rate": flagged_steps / T,
        "trigger_rate": triggered_steps / T,
    }


if __name__ == "__main__":
    out = run_replay(seed=0, T=200, attack_start=100, replay_len=30)
    print("Detected at:", out["detected_at"], "delay:", out["delay"])
    print("flag_rate:", f"{out['flag_rate']:.3f}", "trigger_rate:", f"{out['trigger_rate']:.3f}")
