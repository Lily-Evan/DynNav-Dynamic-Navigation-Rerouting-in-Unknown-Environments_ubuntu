import csv
import numpy as np
from collections import deque
from ukf_fusion import UKF


def log_replay_csv(
    out_csv="ids_replay_log.csv",
    seed=0,
    T=200,
    attack_start=100,
    replay_len=30,
):
    np.random.seed(seed)
    ukf = UKF()

    x_true = np.array([0.0, 0.0, 0.0], dtype=float)
    dt = 1.0
    u = np.array([0.05, 0.02, 0.01], dtype=float)

    buf = deque(maxlen=replay_len)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "attack", "d2", "thr", "ratio", "scale", "flagged", "streak", "triggered", "alert"])

        for t in range(T):
            x_true = x_true + np.array([0.05, 0.02, 0.01])
            ukf.predict(u, dt)

            z_fresh = x_true + np.random.multivariate_normal(np.zeros(3), ukf.R_vo_nominal)

            if t < attack_start:
                buf.append(z_fresh.copy())
                z_vo = z_fresh
                attack = 0
            else:
                attack = 1
                if len(buf) == 0:
                    z_vo = z_fresh
                else:
                    idx = (t - attack_start) % len(buf)
                    z_vo = list(buf)[idx]

            yaw_meas = float(x_true[2] + np.random.normal(0.0, np.sqrt(ukf.R_imu_nominal[0, 0])))

            ukf.update_vo(z_vo)
            ukf.update_imu(yaw_meas)

            info = ukf.last_vo_ids
            d2 = float(info["d2"])
            thr = float(info["thr"])
            ratio = d2 / max(1e-9, thr)
            scale = float(getattr(ukf, "vo_scale_last", 1.0))

            w.writerow([
                t, attack,
                f"{d2:.6f}", f"{thr:.6f}", f"{ratio:.6f}", f"{scale:.6f}",
                int(info["flagged"]), int(info["streak"]), int(info["triggered"]),
                int(bool(ukf.security_alert))
            ])

    print("Wrote:", out_csv)


if __name__ == "__main__":
    log_replay_csv()
