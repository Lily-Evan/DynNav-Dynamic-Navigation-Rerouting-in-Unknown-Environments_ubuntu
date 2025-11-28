class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0

    def compute(self, setpoint, current, dt):
        error = setpoint - current
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0

        out = self.kp*error + self.ki*self.integral + self.kd*derivative
        self.last_error = error
        return out
