from dataclasses import dataclass

@dataclass
class RiskBudgetTracker:
    """Tracks remaining risk budget (sum-risk model)."""
    B_total: float
    B_remaining: float = None
    spent: float = 0.0

    def __post_init__(self):
        self.reset(self.B_total)

    def reset(self, B_total: float):
        self.B_total = float(B_total)
        self.B_remaining = float(B_total)
        self.spent = 0.0

    def consume(self, step_risk: float):
        r = float(step_risk)
        self.spent += r
        self.B_remaining -= r

    def will_violate(self, add_risk: float) -> bool:
        return (self.B_remaining - float(add_risk)) < 0.0

    def remaining(self) -> float:
        return float(self.B_remaining)

