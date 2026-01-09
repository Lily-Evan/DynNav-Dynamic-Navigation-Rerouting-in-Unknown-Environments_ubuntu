from collections import deque
from dataclasses import dataclass


@dataclass
class LearningState:
    learned_lambda: float = 0.7
    drift_threshold: float = 0.6
    failure_threshold: float = 0.4


class ContinualLearner:
    """
    Continual Self-Learning Layer

    - Keeps memory of recent robot reliability
    - Adjusts thresholds dynamically
    - Auto-tunes lambda (risk sensitivity)
    """

    def __init__(
        self,
        history_size: int = 20,
        lambda_step: float = 0.05,
        threshold_step: float = 0.03,
    ):
        self.history_size = history_size
        self.history = deque(maxlen=history_size)

        self.lambda_step = lambda_step
        self.threshold_step = threshold_step

        self.state = LearningState()

    def record_cycle(self, metrics, self_healing_triggered: bool):
        """
        metrics: dict with
          drift
          failure_rate
          heuristic_regret
        """

        entry = dict(
            drift=metrics["drift"],
            failure=metrics["failure_rate"],
            regret=metrics["heuristic_regret"],
            problem=self_healing_triggered,
        )

        self.history.append(entry)

    def learn(self):
        if len(self.history) < 3:
            return self.state

        failures = sum(1 for h in self.history if h["failure"] > 0.4)
        serious_problems = sum(1 for h in self.history if h["problem"])
        avg_drift = sum(h["drift"] for h in self.history) / len(self.history)

        # === λ AUTO-TUNING ===
        if serious_problems > len(self.history) * 0.3:
            self.state.learned_lambda = min(1.0, self.state.learned_lambda + self.lambda_step)
        else:
            self.state.learned_lambda = max(0.3, self.state.learned_lambda - self.lambda_step)

        # === DRIFT THRESHOLD LEARNING ===
        if avg_drift > 0.5:
            self.state.drift_threshold = max(0.3, self.state.drift_threshold - self.threshold_step)
        else:
            self.state.drift_threshold = min(0.8, self.state.drift_threshold + self.threshold_step)

        # === FAILURE THRESHOLD LEARNING ===
        if failures > len(self.history) * 0.25:
            self.state.failure_threshold = max(0.2, self.state.failure_threshold - self.threshold_step)
        else:
            self.state.failure_threshold = min(0.6, self.state.failure_threshold + self.threshold_step)

        return self.state
