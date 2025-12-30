# ================================================================
# Multi-Robot Cooperative Planning with Shared Uncertainty
# ================================================================

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

from risk_explainer import PathStats, choose_best_path, explain_preference


@dataclass
class RobotPathEstimate:
    """
    Εκτίμηση ενός robot για μία συγκεκριμένη διαδρομή.
    """
    robot_name: str
    path_name: str
    drift_exposure: float
    uncertainty_exposure: float


@dataclass
class Robot:
    name: str
    path_estimates: Dict[str, RobotPathEstimate]


def fuse_path_estimates(
    estimates: List[RobotPathEstimate],
    mode: str = "average",
) -> PathStats:
    """
    Fusion πολλαπλών εκτιμήσεων drift/uncertainty για μία path
    σε ένα κοινό "συλλογικό" PathStats.
    """
    if len(estimates) == 0:
        raise ValueError("No estimates provided for fusion")

    path_name = estimates[0].path_name
    # εδώ θεωρούμε ότι το nominal μήκος της διαδρομής είναι ίδιο
    # και έρχεται απ' έξω, οπότε το επιστρέφουμε ως placeholder
    # και θα το συμπληρώσουμε στο demo.
    fused_drift = float(np.mean([e.drift_exposure for e in estimates]))
    fused_var = float(np.mean([e.uncertainty_exposure for e in estimates]))

    # length θα μπει εκ των υστέρων
    return PathStats(
        name=path_name,
        length=0.0,  # θα το ορίσουμε στο ανώτερο επίπεδο
        drift_exposure=fused_drift,
        uncertainty_exposure=fused_var,
    )


def cooperative_choose_path(
    robots: List[Robot],
    base_paths: List[PathStats],
    lambda_risk: float = 1.0,
):
    """
    Κάθε robot έχει εκτιμήσεις per path.
    Συγχωνεύουμε τις εκτιμήσεις για κάθε path και βρίσκουμε
    ποια διαδρομή είναι καλύτερη για το "συλλογικό" σύστημα.
    """
    # index base lengths by path name
    base_lengths = {p.name: p.length for p in base_paths}

    fused_paths: List[PathStats] = []

    # Για κάθε path name μαζεύουμε estimates από όλα τα robots
    path_names = [p.name for p in base_paths]

    for pname in path_names:
        per_robot_estimates: List[RobotPathEstimate] = []
        for r in robots:
            if pname in r.path_estimates:
                per_robot_estimates.append(r.path_estimates[pname])

        if not per_robot_estimates:
            continue

        fused = fuse_path_estimates(per_robot_estimates)
        fused.length = base_lengths[pname]
        fused_paths.append(fused)

    if not fused_paths:
        raise RuntimeError("No fused paths available for cooperative decision.")

    # Επιλογή καλύτερης fused διαδρομής
    best = choose_best_path(fused_paths, lambda_risk=lambda_risk)
    return best, fused_paths
