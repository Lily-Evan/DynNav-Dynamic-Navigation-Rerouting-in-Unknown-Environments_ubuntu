"""
Learning-Ready Language Safety Policy

Αυτό το module φτιάχνει ένα "frontend" για language safety.

Τώρα: χρησιμοποιεί εσωτερικά το rule-based LanguageSafetyPolicy.
Μετά: μπορείς να αντικαταστήσεις την υλοποίηση με πραγματικό NLP μοντέλο
χωρίς να αλλάξεις τον υπόλοιπο κώδικα.
"""

from typing import Dict, List, Optional

from language_safety_policy import LanguageSafetyPolicy, LanguageSafetyDecision


class LearningLanguageSafetyPolicy:
    def __init__(self):
        # προς το παρόν, απλά τυλίγουμε το rule-based
        self._rule_based = LanguageSafetyPolicy()

        # εδώ αργότερα μπορείς να φορτώνεις ML model, vectorizer κτλ.
        self._model = None

    def evaluate(self, message: str) -> LanguageSafetyDecision:
        """
        Στο μέλλον:
          - αν υπάρχει trained model -> χρήση αυτού
          - αλλιώς -> fallback στο rule-based
        """
        if self._model is None:
            return self._rule_based.evaluate(message)
        else:
            # placeholder: ML inference
            # probs = self._model.predict_proba(message_features)
            # εδώ θα χαρτογραφείς τις πιθανότητες σε risk / uncertainty
            return self._rule_based.evaluate(message)

    def fit_dummy(self, examples: List[Dict[str, str]]) -> None:
        """
        Placeholder API για να "εκπαιδεύσεις" αργότερα ένα ML μοντέλο.

        examples: λίστα από dicts με:
            {
              "text": "...",
              "label_risk": float,
              "label_uncertainty": float
            }

        Τώρα δεν κάνει τίποτα (no-op), απλά υπάρχει το API.
        """
        # εδώ αργότερα θα μπει training code
        pass
