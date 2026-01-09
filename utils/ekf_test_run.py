import numpy as np
import matplotlib.pyplot as plt
from ekf_fusion import EKF2D
import math
import random

def simulate_ground_truth(t):
    """
    "Πραγματική" τροχιά: μια ομαλή καμπύλη στο 2D.
    """
    x = 2 + 0.05 * t
    y = 2 + 0.02 * t + 0.5 * math.sin(0.05 * t)
    theta = 0.1 * math.sin(0.03 * t)
    return x, y, theta

def simulate_wheel_odometry(v_true, w_true):
    """
    Προσθέτουμε θόρυβο στη wheel odometry.
    """
    v = v_true + random.gauss(0, 0.02)
    w = w_true + random.gauss(0, 0.01)
    return v, w

def simulate_vo(x_true, y_true, theta_true):
    """
    Θορυβώδης VO μέτρηση.
    """
    x = x_true + random.gauss(0, 0.2)
    y = y_true + random.gauss(0, 0.2)
    theta = theta_true + random.gauss(0, 0.05)
    return x, y, theta

def simulate_imu(theta_true):
    """
    Θορυβώδης IMU yaw.
    """
    return theta_true + random.gauss(0, 0.03)

ekf = EKF2D()

gt_x = []
gt_y = []
vo_x = []
vo_y = []
fused_x = []
fused_y = []

dt = 0.1

prev_x = None
prev_y = None

for t in range(1, 400):

    # Ground truth
    x_t, y_t, th_t = simulate_ground_truth(t)
    gt_x.append(x_t)
    gt_y.append(y_t)

    # VO noisy measurement
    x_vo, y_vo, th_vo = simulate_vo(x_t, y_t, th_t)
    vo_x.append(x_vo)
    vo_y.append(y_vo)

    # IMU yaw
    th_imu = simulate_imu(th_t)

    # Προσέγγιση πραγματικών ταχυτήτων για odometry
    if prev_x is None:
        v_true = 0.05
    else:
        dx = x_t - prev_x
        dy = y_t - prev_y
        v_true = math.sqrt(dx*dx + dy*dy) / dt

    # μικρή προσέγγιση για w_true (εδώ απλά για demo)
    w_true = (th_t - th_t * 0.95) / dt

    prev_x = x_t
    prev_y = y_t

    # Wheel odometry με θόρυβο
    v, w = simulate_wheel_odometry(v_true, w_true)

    # "ψεύτικο" VO confidence: ταλαντώνεται μεταξύ 0.2 και 1.0
    c_vo = 0.2 + 0.8 * (0.5 + 0.5 * math.sin(0.02 * t))

    ekf.set_vo_confidence(c_vo)

    # EKF steps
    ekf.predict(v, w, dt)
    ekf.update_vo(x_vo, y_vo, th_vo)
    ekf.update_imu_yaw(th_imu)

    xf, yf, _ = ekf.get_state()
    fused_x.append(xf)
    fused_y.append(yf)

# PLOTTING
plt.figure()
plt.plot(gt_x, gt_y, 'g-', label='Ground Truth')
plt.plot(vo_x, vo_y, 'r.', alpha=0.4, label='VO Measurements')
plt.plot(fused_x, fused_y, 'b-', linewidth=2, label='EKF Fused')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.title("EKF Fusion: VO + Odom + IMU (with VO confidence)")
plt.savefig("ekf_fusion_result.png", dpi=200)
plt.show()
