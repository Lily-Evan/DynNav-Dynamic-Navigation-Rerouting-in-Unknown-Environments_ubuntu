from dataclasses import dataclass
from typing import Optional, Dict, Any

from self_healing_policy import SelfHealingPolicy, SelfHealingConfig
from learning_language_safety import LearningLanguageSafetyPolicy
from trust_layer import TrustManager, TrustState
from ethical_layer import EthicalRiskPolicy
from continual_learning import ContinualLearner


@dataclass
class RobotMetrics:
    """
    Normalized robot reliability metrics in [0, 1].
    These can come from localization, planning, or higher-level monitors.
    """
    drift: float = 0.0
    calibration_error: float = 0.0
    heuristic_regret: float = 0.0
    failure_rate: float = 0.0


class NavigationController:
    """
    High-level navigation supervisor that integrates:

    - Self-Healing Navigation
    - Language-Driven Safety (learning-ready frontend)
    - Ethical Risk Modulation
    - Trust Dynamics
    - Continual Self-Learning (adaptive thresholds + lambda)

    It does NOT do planning itself; instead, it updates the parameters
    of an external planner object (e.g., AbstractPlanner).
    """

    def __init__(self, planner: Any):
        self.planner = planner

        # Core policies
        self.self_healing = SelfHealingPolicy(SelfHealingConfig())
        self.lang_safety = LearningLanguageSafetyPolicy()
        self.trust_manager = TrustManager()
        self.trust_state = TrustState()
        self.ethical_policy = EthicalRiskPolicy()
        self.learner = ContinualLearner()

        self.step = 0

    def update(self, metrics: RobotMetrics, human_message: Optional[str]) -> Dict[str, Any]:
        """
        One navigation control cycle:

        1. Evaluate self-healing based on robot reliability metrics.
        2. Interpret human language as safety and ethical constraints.
        3. Update trust according to system behaviour and human input.
        4. Apply trust-, ethics- and language-weighted risk to the planner.
        5. Perform continual learning on thresholds and lambda.

        Returns a dictionary with all intermediate reasoning for logging / analysis.
        """
        self.step += 1

        # ------------------------------------------------------------------
        # 1) SELF-HEALING
        # ------------------------------------------------------------------
        sh_decision = self.self_healing.evaluate(
            step=self.step,
            metrics=dict(
                drift=metrics.drift,
                calibration_error=metrics.calibration_error,
                heuristic_regret=metrics.heuristic_regret,
                failure_rate=metrics.failure_rate,
            ),
        )
        sh_dict = sh_decision.to_dict()

        # ------------------------------------------------------------------
        # 2) LANGUAGE SAFETY (learning-ready frontend)
        # ------------------------------------------------------------------
        if human_message:
            lang_decision = self.lang_safety.evaluate(human_message)
            lang_dict = lang_decision.to_dict()
        else:
            lang_decision = None
            lang_dict = None

        # ------------------------------------------------------------------
        # 3) ETHICAL LAYER (children, elderly, crowding → extra ethical risk)
        # ------------------------------------------------------------------
        if lang_dict is not None:
            ethical_decision = self.ethical_policy.evaluate_from_language(lang_dict)
            ethical_info = dict(
                ethical_risk_scale=ethical_decision.ethical_risk_scale,
                ethical_factors=ethical_decision.ethical_factors,
            )
            ethical_scale = ethical_decision.ethical_risk_scale
        else:
            ethical_decision = None
            ethical_info = None
            ethical_scale = 1.0

        # ------------------------------------------------------------------
        # 4) TRUST DYNAMICS
        # ------------------------------------------------------------------
        self.trust_state = self.trust_manager.update(
            trust=self.trust_state,
            self_healing_decision=sh_dict,
            language_decision=None if lang_dict is None else lang_dict,
        )

        # Base risk / uncertainty from language (or neutral)
        if lang_dict is not None:
            base_risk_scale = float(lang_dict["risk_scale"])
            uncertainty_scale = float(lang_dict["uncertainty_scale"])
        else:
            base_risk_scale = 1.0
            uncertainty_scale = 1.0

        # Add ethical amplification
        risk_with_ethics = base_risk_scale * ethical_scale

        # Trust-weighted risk (lower trust → more conservative)
        effective_risk_scale = self.trust_manager.compute_trust_weighted_risk(
            base_risk_scale=risk_with_ethics,
            trust=self.trust_state,
        )





        # ------------------------------------------------------------------
        # 5) CONTINUAL SELF-LEARNING (thresholds + lambda evolution)
        # ------------------------------------------------------------------
        self.learner.record_cycle(
            metrics=dict(
                drift=metrics.drift,
                failure_rate=metrics.failure_rate,
                heuristic_regret=metrics.heuristic_regret,
            ),
            self_healing_triggered=sh_decision.trigger,
        )
        learning_state = self.learner.learn()

        # Optional: use learned thresholds / lambda to update self-healing config
        # (simple coupling; you can refine this policy)
        self.self_healing.config.drift_threshold = learning_state.drift_threshold
        self.self_healing.config.failure_threshold = learning_state.failure_threshold
        # Blend learned lambda with self-healing lambda (e.g., average)
        blended_lambda = 0.5 * sh_decision.new_lambda + 0.5 * learning_state.learned_lambda

        # ------------------------------------------------------------------
        # 6) APPLY TO PLANNER
        # ------------------------------------------------------------------
        self.planner.update_parameters(
            risk_scale=effective_risk_scale,
            uncertainty_scale=uncertainty_scale,
            lambda_risk=blended_lambda,
            safe_mode=sh_decision.safe_mode,
        )

        # ------------------------------------------------------------------
        # 7) RETURN DIAGNOSTICS
        # ------------------------------------------------------------------
        return dict(
            self_healing=sh_dict,
            language_safety=None if lang_decision is None else lang_decision.to_dict(),
            ethical=ethical_info,
            trust=dict(value=self.trust_state.value),
            learning_state=dict(
                learned_lambda=learning_state.learned_lambda,
                drift_threshold=learning_state.drift_threshold,
                failure_threshold=learning_state.failure_threshold,
            ),
            planner_state=self.planner.debug_state(),
        )
