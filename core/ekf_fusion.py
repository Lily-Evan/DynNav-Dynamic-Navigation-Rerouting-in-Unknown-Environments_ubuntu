import numpy as np
import math

class EKF2D:
    def __init__(self):
        # State: [x, y, theta]
        self.x = np.zeros((3, 1))       # αρχική εκτίμηση
        self.P = np.eye(3) * 0.1        # covariance

        # Process noise (θόρυβος μοντέλου κίνησης)
        self.Q = np.diag([0.05, 0.05, 0.02])

        # Measurement noise VO (θα αλλάζει δυναμικά με confidence)
        self.R_vo = np.diag([0.1, 0.1, 0.05])

        # IMU yaw measurement noise
        self.R_imu = np.array([[0.02]])

    def predict(self, v, w, dt):
        """
        Prediction model using wheel odometry:
        v = linear velocity
        w = angular velocity
        dt = timestep
        """

        theta = self.x[2, 0]

        # Μη γραμμικό μοντέλο κίνησης διαφορικού ρομπότ
        if abs(w) < 1e-6:
            dx = v * dt * math.cos(theta)
            dy = v * dt * math.sin(theta)
            dtheta = 0.0
        else:
            dx = (v / w) * (math.sin(theta + w * dt) - math.sin(theta))
            dy = (v / w) * (-math.cos(theta + w * dt) + math.cos(theta))
            dtheta = w * dt

        # Ενημέρωση κατάστασης
        self.x[0, 0] += dx
        self.x[1, 0] += dy
        self.x[2, 0] += dtheta

        # Jacobian ως προς την κατάσταση (F)
        F = np.eye(3)
        F[0, 2] = -v * dt * math.sin(theta)
        F[1, 2] =  v * dt * math.cos(theta)

        # Ενημέρωση covariance
        self.P = F @ self.P @ F.T + self.Q

    def set_vo_confidence(self, c):
        """
        c in [0, 1]  (0 = πολύ κακή VO μέτρηση, 1 = πολύ καλή)
        Προσαρμόζουμε το R_vo ανάλογα.
        """
        c = max(0.0, min(1.0, float(c)))  # clamp [0,1]

        # base noise (κακή περίπτωση) - μεγαλύτερο
        base = np.diag([0.5, 0.5, 0.2])

        # best noise (καλή περίπτωση) - μικρότερο
        best = np.diag([0.05, 0.05, 0.02])

        # linear interpolation: R = (1-c)*base + c*best
        self.R_vo = (1.0 - c) * base + c * best

    def update_vo(self, x_vo, y_vo, theta_vo):
        """
        Update using VO pose measurement.
        z = [x_vo, y_vo, theta_vo]
        """
        z = np.array([[x_vo], [y_vo], [theta_vo]])

        # Measurement model: h(x) = x (identity)
        H = np.eye(3)

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R_vo
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state & covariance
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

    def update_imu_yaw(self, yaw):
        """
        IMU yaw measurement (μόνο γωνία)
        """
        z = np.array([[yaw]])
        H = np.array([[0.0, 0.0, 1.0]])  # μετράμε μόνο θ

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

    def get_state(self):
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])
