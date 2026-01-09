import csv
import os


class ResultsLogger:
    """
    Logs navigation intelligence state evolution over time
    into CSV for scientific analysis and plotting.
    """

    def __init__(self, filename="navigation_results.csv"):
        self.filename = filename
        self.initialized = False

    def _init_file(self):
        if os.path.exists(self.filename):
            self.initialized = True
            return

        with open(self.filename, mode="w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "step",

                # Robot metrics
                "drift",
                "failure_rate",
                "heuristic_regret",

                # Self Healing
                "self_healing_trigger",
                "safe_mode",

                # Language Risk
                "risk_scale",
                "uncertainty_scale",

                # Ethical
                "ethical_scale",

                # Trust
                "trust",

                # Continual Learning
                "learned_lambda",
                "drift_threshold",
                "failure_threshold",

                # Planner
                "linear_velocity",
                "angular_velocity",
                "inflation_radius",
                "lambda_used",
            ])

        self.initialized = True

    def log(self, step, metrics, result):
        if not self.initialized:
            self._init_file()

        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                step,

                metrics.drift,
                metrics.failure_rate,
                metrics.heuristic_regret,

                result["self_healing"]["SELF_HEALING_TRIGGER"],
                result["self_healing"]["safe_mode"],

                result["language_safety"]["risk_scale"] if result["language_safety"] else 1.0,
                result["language_safety"]["uncertainty_scale"] if result["language_safety"] else 1.0,

                result["ethical"]["ethical_risk_scale"] if result["ethical"] else 1.0,

                result["trust"]["value"],

                result["learning_state"]["learned_lambda"],
                result["learning_state"]["drift_threshold"],
                result["learning_state"]["failure_threshold"],

                result["planner_state"]["max_linear_velocity"],
                result["planner_state"]["max_angular_velocity"],
                result["planner_state"]["obstacle_inflation_radius"],
                result["planner_state"]["lambda_risk"],
            ])
