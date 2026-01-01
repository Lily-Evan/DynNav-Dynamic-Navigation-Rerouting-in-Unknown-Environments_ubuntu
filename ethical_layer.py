from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EthicalDecision:
    ethical_risk_scale: float
    ethical_factors: List[str]


class EthicalRiskPolicy:
    """
    Ethical / social-aware risk modulation.

    Χρησιμοποιεί τα language factors (children, elderly, crowding)
    και τα μετατρέπει σε επιπλέον ethical_risk_scale.
    """

    def __init__(self, base_scale: float = 1.0, max_scale: float = 2.0):
        self.base_scale = base_scale
        self.max_scale = max_scale

    def evaluate_from_language(self, language_decision: Dict) -> EthicalDecision:
        factors: List[str] = language_decision.get("factors", [])
        scale = self.base_scale

        # απλοί κανόνες:
        for f in factors:
            lf = f.lower()
            if "children" in lf:
                scale *= 1.3
            if "elderly" in lf:
                scale *= 1.4
            if "crowding" in lf or "many people" in lf:
                scale *= 1.2

        scale = min(scale, self.max_scale)
        return EthicalDecision(ethical_risk_scale=scale, ethical_factors=factors)
