import numpy as np

class EKF:
    def __init__(self):
        # state: [x, y, yaw]
        self.x = np.zeros(3)

        # covariance
        self.P = np.eye(3) * 0.1

        # process noise
        self.Q = np.diag([0.05, 0.05, 0.01])

        # measurement noise
        self.R_vo = np.diag([0.1, 0.1, 0.05])
        self.R_imu = np.diag([0.01])
        self.R_odom = np.diag([0.05, 0.05])

    def predict(self, u, dt):
        # u = [v, omega]
        v, w = u

        x, y, yaw = self.x

        # motion model
        x_new = x + v * np.cos(yaw) * dt
        y_new = y + v * np.sin(yaw) * dt
        yaw_new = yaw + w * dt

        self.x = np.array([x_new, y_new, yaw_new])

        # Jacobian
        F = np.array([
            [1, 0, -v*np.sin(yaw)*dt],
            [0, 1,  v*np.cos(yaw)*dt],
            [0, 0,  1]
        ])

        self.P = F @ self.P @ F.T + self.Q

    def update_vo(self, z):
        # z = [x, y, yaw]
        H = np.eye(3)
        R = self.R_vo

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

    def update_imu(self, yaw_meas):
        # imu measures only yaw
        H = np.array([[0, 0, 1]])
        R = self.R_imu

        y = yaw_meas - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += (K @ y).flatten()
        self.P = (np.eye(3) - K @ H) @ self.P

    def update_odom(self, z):
        # odom = [x, y]
        H = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        R = self.R_odom

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += (K @ y)
        self.P = (np.eye(3) - K @ H) @ self.P

    def get_state(self):
        return self.x, self.P

