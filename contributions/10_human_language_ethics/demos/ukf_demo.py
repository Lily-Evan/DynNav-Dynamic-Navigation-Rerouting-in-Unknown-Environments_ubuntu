import numpy as np
import matplotlib.pyplot as plt

from ukf_fusion import UKF  # εισάγουμε την κλάση UKF

def simulate_trajectory(T=200, dt=0.1):
    """
    Προσομοιώνει μια απλή τροχιά σε κύκλο:
    state = [x, y, yaw]
    """
    xs, ys, yaws = [], [], []
    x, y, yaw = 0.0, 0.0, 0.0
    v = 0.1          # m/s
    yaw_rate = 0.05  # rad/s

    for k in range(T):
        # Ενημέρωση ground truth
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw += yaw_rate * dt

        xs.append(x)
        ys.append(y)
        yaws.append(yaw)

    return np.array(xs), np.array(ys), np.array(yaws)

if __name__ == "__main__":
    dt = 0.1
    T = 200

    # Ground truth τροχιά
    xs_true, ys_true, yaws_true = simulate_trajectory(T=T, dt=dt)

    # Δημιουργία noisy μετρήσεων VO (pose) και IMU (yaw)
    vo_x = xs_true + np.random.normal(0, 0.05, size=T)
    vo_y = ys_true + np.random.normal(0, 0.05, size=T)
    vo_yaw = yaws_true + np.random.normal(0, 0.02, size=T)

    imu_yaw = yaws_true + np.random.normal(0, 0.01, size=T)

    # "Έλεγχος" από odometry (v_x, v_y, yaw_rate) = true + μικρός θόρυβος
    vx = np.diff(xs_true, prepend=xs_true[0]) / dt
    vy = np.diff(ys_true, prepend=ys_true[0]) / dt
    yaw_rate_cmd = np.diff(yaws_true, prepend=yaws_true[0]) / dt

    vx += np.random.normal(0, 0.01, size=T)
    vy += np.random.normal(0, 0.01, size=T)
    yaw_rate_cmd += np.random.normal(0, 0.005, size=T)

    # UKF
    ukf = UKF()
    x_est, y_est, yaw_est = [], [], []

    for k in range(T):
        u = [vx[k], vy[k], yaw_rate_cmd[k]]
        ukf.predict(u=u, dt=dt)

        # ενημέρωση με VO pose
        z_vo = np.array([vo_x[k], vo_y[k], vo_yaw[k]])
        ukf.update_vo(z_vo)

        # ενημέρωση με IMU yaw
        ukf.update_imu(imu_yaw[k])

        x_est.append(ukf.x[0])
        y_est.append(ukf.x[1])
        yaw_est.append(ukf.x[2])

    x_est = np.array(x_est)
    y_est = np.array(y_est)

    # ============ PLOTS ============
    plt.figure(figsize=(7, 6))

    # Τροχιές XY
    plt.plot(xs_true, ys_true, label="Ground Truth", linewidth=2)
    plt.plot(vo_x, vo_y, ".", alpha=0.5, label="Noisy VO", markersize=3)
    plt.plot(x_est, y_est, label="UKF estimate", linewidth=2)

    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("UKF Fusion Demo (VO + Odom + IMU)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("ukf_demo_result.png", dpi=200)
    plt.show()

    print("UKF demo completed. Saved ukf_demo_result.png")
