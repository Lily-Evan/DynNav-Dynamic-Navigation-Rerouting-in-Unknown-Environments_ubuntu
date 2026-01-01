from dataclasses import dataclass
from typing import Optional, Dict, Any

from self_healing_policy import SelfHealingPolicy, SelfHealingConfig
from learning_language_safety import LearningLanguageSafetyPolicy
from trust_layer import TrustManager, TrustState
from ethical_layer import EthicalRiskPolicy


@dataclass
class RobotMetrics:
    drift: float = 0.0
    calibration_error: float = 0.0
    heuristic_regret: float = 0.0
    failure_rate: float = 0.0


class NavigationController:
    def __init__(self, planner):
        self.planner = planner

        self.self_healing = SelfHealingPolicy(SelfHealingConfig())
        self.lang_safety = LearningLanguageSafetyPolicy()
        self.trust_manager = TrustManager()
        self.trust_state = TrustState()
        self.ethical_policy = EthicalRiskPolicy()

        self.step = 0

    def update(self, metrics: RobotMetrics, human_message: Optional[str]) -> Dict[str, Any]:
        self.step += 1

        # 1) Self-Healing
        sh = self.self_healing.evaluate(
            step=self.step,
            metrics=dict(
                drift=metrics.drift,
                calibration_error=metrics.calibration_error,
                heuristic_regret=metrics.heuristic_regret,
                failure_rate=metrics.failure_rate,
            ),
        )

        # 2) Language Safety (learning-ready wrapper)
        if human_message:
            lang_decision = self.lang_safety.evaluate(human_message)
            lang_dict = lang_decision.to_dict()
        else:
            lang_decision = None
            lang_dict = None

        # 3) Ethical Layer (με βάση language factors)
        if lang_dict is not None:
            ethical_decision = self.ethical_policy.evaluate_from_language(lang_dict)
            ethical_scale = ethical_decision.ethical_risk_scale
        else:
            ethical_decision = None
            ethical_scale = 1.0

        # 4) Trust Layer
        self.trust_state = self.trust_manager.update(
            trust=self.trust_state,
            self_healing_decision=sh.to_dict(),
            language_decision=None if lang_dict is None else lang_dict,
        )

        # base risk από language
        if lang_dict is not None:
            base_risk_scale = float(lang_dict["risk_scale"])
            uncertainty_scale = float(lang_dict["uncertainty_scale"])
        else:
            base_risk_scale = 1.0
            uncertainty_scale = 1.0

        # ethical risk
        risk_with_ethics = base_risk_scale * ethical_scale

        # trust-weighted risk
        effective_risk_scale = self.trust_manager.compute_trust_weighted_risk(
            base_risk_scale=risk_with_ethics,
            trust=self.trust_state,
        )

        # 5) Εφαρμογή στον planner
        self.planner.update_parameters(
            risk_scale=effective_risk_scale,
            uncertainty_scale=uncertainty_scale,
            lambda_risk=sh.new_lambda,
            safe_mode=sh.safe_mode,
        )

        # 🔴 ΕΔΩ είναι το σημαντικό return με ΟΛΑ τα keys
        return dict(
            self_healing=sh.to_dict(),
            language_safety=None if lang_decision is None else lang_decision.to_dict(),
            ethical=None
            if ethical_decision is None
            else dict(
                ethical_risk_scale=ethical_decision.ethical_risk_scale,
                ethical_factors=ethical_decision.ethical_factors,
            ),
            trust=dict(value=self.trust_state.value),
            planner_state=self.planner.debug_state(),
        )
