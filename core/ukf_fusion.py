import numpy as np
from security_monitor import InnovationMonitor, MonitorConfig


class UKF:
    def __init__(self):
        # State: [x, y, yaw]
        self.x = np.zeros(3, dtype=float)
        self.P = np.eye(3, dtype=float) * 0.1

        # Measurement noise (nominal)
        self.R_vo = np.diag([0.05, 0.05, 0.02])
        self.R_imu = np.array([[0.01]])

        self.R_vo_nominal = self.R_vo.copy()
        self.R_imu_nominal = self.R_imu.copy()

        # Minimal process noise
        self.Q = np.eye(3, dtype=float) * 1e-3


        cfg_vo = MonitorConfig(alpha=0.01, window=200, warmup=20, mode="kofn", k=3, n=10)
        cfg_imu = MonitorConfig(alpha=0.01, window=200, warmup=20, mode="consecutive", consecutive=3)

        self.mon_vo = InnovationMonitor(cfg_vo)
        self.mon_imu = InnovationMonitor(cfg_imu)


        # Security state
        self.security_alert = False
        self.last_vo_ids = None
        self.last_imu_ids = None

        # Adaptive scales (for logging/plots)
        self.vo_scale_last = 1.0
        self.imu_scale_last = 1.0

    def predict(self, u, dt):
        """
        Minimal predict model:
          u = [vx, vy, yaw_rate] in world frame
        """
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.shape[0] != 3:
            raise ValueError("u must be [vx, vy, yaw_rate]")

        vx, vy, yaw_rate = float(u[0]), float(u[1]), float(u[2])

        self.x[0] += vx * dt
        self.x[1] += vy * dt
        self.x[2] += yaw_rate * dt

        self.P = self.P + self.Q * dt

    def _adaptive_scale(self, d2: float, thr: float, k: float = 4.0, scale_max: float = 50.0) -> float:
        ratio = float(d2) / max(1e-9, float(thr))
        if ratio <= 1.0:
            return 1.0
        return float(min(scale_max, 1.0 + k * (ratio - 1.0)))

    def update_vo(self, z):
        z = np.asarray(z, dtype=float).reshape(3)
        H = np.eye(3)
        S = H @ self.P @ H.T + self.R_vo
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ self.x

        # IDS: innovation gating
        info = self.mon_vo.update(y, S, meta={"sensor": "vo"})
        self.last_vo_ids = info

        # Adaptive trust weighting
        self.vo_scale_last = self._adaptive_scale(info["d2"], info["thr"], k=4.0, scale_max=50.0)
        self.R_vo = self.R_vo_nominal * self.vo_scale_last

        # Trigger => sticky alert (fail-safe latch)
        if info["triggered"]:
            self.security_alert = True

        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

    def update_imu(self, yaw_meas):
        yaw_meas = float(yaw_meas)
        H = np.array([[0.0, 0.0, 1.0]])
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)
        y = yaw_meas - float(H @ self.x)

        info = self.mon_imu.update(np.array([y], dtype=float), S, meta={"sensor": "imu"})
        self.last_imu_ids = info

        self.imu_scale_last = self._adaptive_scale(info["d2"], info["thr"], k=4.0, scale_max=50.0)
        self.R_imu = self.R_imu_nominal * self.imu_scale_last

        if info["triggered"]:
            self.security_alert = True

        self.x = self.x + (K.flatten() * y)
        self.P = (np.eye(3) - K @ H) @ self.P
