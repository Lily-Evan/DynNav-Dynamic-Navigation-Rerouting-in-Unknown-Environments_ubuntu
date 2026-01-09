class AbstractPlanner:
    def __init__(self):
        self.max_linear_velocity = 0.6
        self.max_angular_velocity = 1.0
        self.obstacle_inflation_radius = 0.4
        self.goal_tolerance = 0.15
        self.lambda_risk = 1.0

    def update_parameters(self, risk_scale, uncertainty_scale, lambda_risk, safe_mode):
        self.obstacle_inflation_radius = min(0.4 * risk_scale, 1.0)

        self.max_linear_velocity = max(0.1, 0.6 / risk_scale)
        self.max_angular_velocity = max(0.2, 1.0 / risk_scale)

        self.goal_tolerance = min(0.15 * uncertainty_scale, 0.5)

        self.lambda_risk = lambda_risk

        if safe_mode:
            self.max_linear_velocity = min(self.max_linear_velocity, 0.2)
            self.max_angular_velocity = min(self.max_angular_velocity, 0.5)
            self.obstacle_inflation_radius = max(self.obstacle_inflation_radius, 0.6)

    def debug_state(self):
        return dict(
            max_linear_velocity=self.max_linear_velocity,
            max_angular_velocity=self.max_angular_velocity,
            obstacle_inflation_radius=self.obstacle_inflation_radius,
            goal_tolerance=self.goal_tolerance,
            lambda_risk=self.lambda_risk,
        )
