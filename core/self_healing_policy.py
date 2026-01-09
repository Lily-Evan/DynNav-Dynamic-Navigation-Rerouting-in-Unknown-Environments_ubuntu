"""
Self-Healing Navigation Policy
--------------------------------
Reference implementation for self-healing autonomy in navigation.

The policy monitors normalized reliability metrics in [0, 1]:

  - drift            : localization drift / VO confidence
  - calibration_error: miscalibration of uncertainty
  - heuristic_regret : deviation vs optimal cost
  - failure_rate     : recent failure / near-miss probability

If metrics exceed configurable thresholds, the policy proposes
self-healing actions, adjusts a risk weight lambda, and can activate
a Safe Mode.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SelfHealingConfig:
    # Thresholds in [0, 1]
    drift_threshold: float = 0.6
    calibration_threshold: float = 0.6
    regret_threshold: float = 0.6
    failure_threshold: float = 0.4

    # Safe mode conditions
    safe_mode_failure_threshold: float = 0.6
    safe_mode_regret_threshold: float = 0.7

    # Lambda (risk-weight) bounds
    min_lambda: float = 0.5
    max_lambda: float = 3.0
    lambda_step: float = 0.25

    # Cooldown in steps between triggers
    cooldown_steps: int = 2


@dataclass
class SelfHealingDecision:
    trigger: bool
    reasons: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    new_lambda: float = 1.0
    safe_mode: bool = False

    def to_dict(self) -> Dict:
        return {
            "SELF_HEALING_TRIGGER": self.trigger,
            "reasons": self.reasons,
            "recommended_actions": self.recommended_actions,
            "new_lambda": self.new_lambda,
            "safe_mode": self.safe_mode,
        }


class SelfHealingPolicy:
    """
    Closed-loop autonomy maintenance policy.

    Usage:

        policy = SelfHealingPolicy()
        decision = policy.evaluate(
            step=7,
            metrics={
                "drift": 0.7,
                "calibration_error": 0.3,
                "heuristic_regret": 0.8,
                "failure_rate": 0.5,
            },
        )
    """

    def __init__(self, config: Optional[SelfHealingConfig] = None):
        self.config = config or SelfHealingConfig()
        self._last_trigger_step: Optional[int] = None
        self._lambda: float = 1.0  # planning risk-weight

    @property
    def current_lambda(self) -> float:
        return self._lambda

    def _cooldown_active(self, step: int) -> bool:
        if self._last_trigger_step is None:
            return False
        return (step - self._last_trigger_step) < self.config.cooldown_steps

    def _maybe_increase_lambda(self) -> None:
        """Move λ towards conservative, safety-oriented planning."""
        self._lambda = min(self._lambda + self.config.lambda_step, self.config.max_lambda)

    def _maybe_decrease_lambda(self) -> None:
        """Optional: relax λ if everything is healthy."""
        self._lambda = max(self._lambda - self.config.lambda_step, self.config.min_lambda)

    def evaluate(self, step: int, metrics: Dict[str, float]) -> SelfHealingDecision:
        """
        Evaluate reliability metrics and decide whether to trigger self-healing.

        Parameters
        ----------
        step : int
            Current planning step or time index.
        metrics : dict
            Normalized metrics in [0, 1] with keys:
              'drift', 'calibration_error', 'heuristic_regret', 'failure_rate'

        Returns
        -------
        SelfHealingDecision
        """
        drift = float(metrics.get("drift", 0.0))
        calib = float(metrics.get("calibration_error", 0.0))
        regret = float(metrics.get("heuristic_regret", 0.0))
        failure = float(metrics.get("failure_rate", 0.0))

        reasons: List[str] = []
        actions: List[str] = []

        # Drift
        if drift >= self.config.drift_threshold:
            reasons.append(f"drift {drift:.2f} ≥ threshold {self.config.drift_threshold:.2f}")
            actions.append("adjust_state_estimator")

        # Calibration error
        if calib >= self.config.calibration_threshold:
            reasons.append(
                "calibration_error "
                f"{calib:.2f} ≥ threshold {self.config.calibration_threshold:.2f}"
            )
            actions.append("recalibrate_uncertainty_models")

        # Heuristic regret
        if regret >= self.config.regret_threshold:
            reasons.append(
                "heuristic_regret "
                f"{regret:.2f} ≥ threshold {self.config.regret_threshold:.2f}"
            )
            actions.append("retrain_heuristic_small_batch")

        # Failure rate
        if failure >= self.config.failure_threshold:
            reasons.append(
                "failure_rate "
                f"{failure:.2f} ≥ threshold {self.config.failure_threshold:.2f}"
            )
            actions.append("review_recent_failures")

        # Safe mode?
        safe_mode = (
            failure >= self.config.safe_mode_failure_threshold
            or regret >= self.config.safe_mode_regret_threshold
        )
        if safe_mode:
            actions.append("activate_safe_mode")
            actions.append("increase_monitoring_frequency")

        trigger = len(reasons) > 0 and not self._cooldown_active(step)

        if trigger:
            self._last_trigger_step = step
            self._maybe_increase_lambda()
        else:
            # If everything is very healthy, we can relax λ slightly
            if max(drift, calib, regret, failure) < 0.2:
                self._maybe_decrease_lambda()

        # Always suggest increased risk weight when issues exist
        if len(reasons) > 0 and "increase_risk_weight" not in actions:
            actions.append("increase_risk_weight")

        decision = SelfHealingDecision(
            trigger=trigger,
            reasons=reasons,
            recommended_actions=actions,
            new_lambda=self._lambda,
            safe_mode=safe_mode,
        )
        return decision
