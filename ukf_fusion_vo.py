import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ukf_fusion import UKF  # κλάση UKF που ήδη έχεις


def load_vo_trajectory(csv_path: str = "vo_trajectory.csv"):
    """
    Φορτώνει την VO τροχιά.
    Χρησιμοποιούμε x, z ως 2D επίπεδο (όπως στα coverage/trajectory plots).
    """
    df = pd.read_csv(csv_path)
    xs = df["x"].to_numpy()
    zs = df["z"].to_numpy()   # z ως 'κάθετη' συντεταγμένη στο επίπεδο
    return xs, zs


def compute_yaw_from_xy(xs: np.ndarray, ys: np.ndarray):
    """
    Υπολογίζει yaw γωνία από διαδοχικές διαφορές (atan2).
    """
    dx = np.diff(xs, prepend=xs[0])
    dy = np.diff(ys, prepend=ys[0])
    yaws = np.arctan2(dy, dx)
    return yaws


def moving_average(x: np.ndarray, window: int = 31):
    """
    Απλό smoothing με moving average (πρέπει window να είναι περιττός).
    """
    if window < 3:
        return x.copy()
    if window % 2 == 0:
        window += 1  # κάν' το περιττό
    kernel = np.ones(window) / window
    # 'same' για να διατηρήσουμε ίδιο μήκος
    return np.convolve(x, kernel, mode="same")


if __name__ == "__main__":
    dt = 0.1   # υποθέτουμε ~10Hz. Άλλαξέ το αν ξέρεις το πραγματικό rate.

    # 1. Φόρτωση VO τροχιάς (x,z)
    xs_vo, ys_vo = load_vo_trajectory("vo_trajectory.csv")
    N = len(xs_vo)
    t = np.arange(N) * dt

    # 2. Smooth VO trajectory για reference (pseudo-ground-truth)
    xs_ref = moving_average(xs_vo, window=31)
    ys_ref = moving_average(ys_vo, window=31)

    # 3. Υπολογισμός yaw από την (μη φιλτραρισμένη) VO τροχιά
    yaws_vo = compute_yaw_from_xy(xs_vo, ys_vo)

    # 4. Δημιουργία pseudo-odometry (vx, vy, yaw_rate) από VO + θόρυβο
    vx = np.diff(xs_vo, prepend=xs_vo[0]) / dt
    vy = np.diff(ys_vo, prepend=ys_vo[0]) / dt
    yaw_rate = np.diff(yaws_vo, prepend=yaws_vo[0]) / dt

    # Προσθέτουμε μικρό θόρυβο για να μοιάζει με wheel odometry
    rng = np.random.default_rng(seed=42)
    vx_odom = vx + rng.normal(0, 0.02, size=N)
    vy_odom = vy + rng.normal(0, 0.02, size=N)
    yaw_rate_odom = yaw_rate + rng.normal(0, 0.01, size=N)

    # 5. "Μετρήσεις" VO (pose) με επιπλέον θόρυβο
    vo_x_meas = xs_vo + rng.normal(0, 0.05, size=N)
    vo_y_meas = ys_vo + rng.normal(0, 0.05, size=N)
    vo_yaw_meas = yaws_vo + rng.normal(0, 0.02, size=N)

    # 6. "Μετρήσεις" IMU yaw με μικρότερο θόρυβο
    imu_yaw = yaws_vo + rng.normal(0, 0.01, size=N)

    # 7. Ολοκλήρωση odom χωρίς VO/IMU (για σύγκριση)
    x_odom = [xs_vo[0]]
    y_odom = [ys_vo[0]]
    yaw_odom = [yaws_vo[0]]
    for k in range(1, N):
        x_odom.append(x_odom[-1] + vx_odom[k] * dt)
        y_odom.append(y_odom[-1] + vy_odom[k] * dt)
        yaw_odom.append(yaw_odom[-1] + yaw_rate_odom[k] * dt)
    x_odom = np.array(x_odom)
    y_odom = np.array(y_odom)

    # 8. UKF fusion (VO + Odom + IMU)
    ukf = UKF()
    x_est, y_est = [], []

    for k in range(N):
        u = [vx_odom[k], vy_odom[k], yaw_rate_odom[k]]
        ukf.predict(u=u, dt=dt)

        z_vo = np.array([vo_x_meas[k], vo_y_meas[k], vo_yaw_meas[k]])
        ukf.update_vo(z_vo)

        ukf.update_imu(imu_yaw[k])

        x_est.append(ukf.x[0])
        y_est.append(ukf.x[1])

    x_est = np.array(x_est)
    y_est = np.array(y_est)

    # 9. Plot σύγκρισης
    plt.figure(figsize=(7, 6))

    # Smoothed VO reference (σαν ground truth)
    plt.plot(xs_ref, ys_ref, color="blue", label="Smoothed VO reference", linewidth=2)

    # Ολοκλήρωση μόνο με odom
    plt.plot(x_odom, y_odom, "--", color="orange", label="Odom-only integration", linewidth=1)

    # Noisy VO measurements
    plt.scatter(vo_x_meas, vo_y_meas, s=6, color="gold", alpha=0.5, label="Noisy VO measurements")

    # UKF fused estimate
    plt.plot(x_est, y_est, color="green", label="UKF fused estimate", linewidth=2)

    plt.axis("equal")
    plt.xlabel("X (VO units)")
    plt.ylabel("Y (VO units)")
    plt.title("UKF Fusion on Real VO Trajectory (x-z projection)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("ukf_fusion_vo_result.png", dpi=200)
    plt.show()

    print("UKF fusion on real VO completed. Saved ukf_fusion_vo_result.png")
