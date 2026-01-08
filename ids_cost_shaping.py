import numpy as np

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def anomaly_to_scale(score: float, alpha: float = 1.5) -> float:
    """
    Convert anomaly score s in [0,1] to multiplicative scale >=1.
    scale = 1 + alpha*s
    """
    s = clamp01(score)
    return 1.0 + alpha * s

def shape_risk_grid(risk_grid: np.ndarray, score: float, alpha: float = 1.5) -> np.ndarray:
    """
    Risk shaping: R' = R * (1 + alpha*s)
    """
    scale = anomaly_to_scale(score, alpha=alpha)
    return risk_grid * scale

def shape_lambda(lambda0: float, score: float, beta: float = 2.0) -> float:
    """
    Lambda shaping: lambda' = lambda0*(1 + beta*s)
    """
    s = clamp01(score)
    return float(lambda0 * (1.0 + beta * s))

def shape_trust(trust0: float, score: float, gamma: float = 0.7) -> float:
    """
    Trust shaping: trust' = trust0*(1 - gamma*s)
    """
    s = clamp01(score)
    return float(max(0.0, trust0 * (1.0 - gamma * s)))

def synthetic_anomaly_stream(T: int, attack_start: int, attack_end: int,
                             base: float = 0.05, attack_level: float = 0.8,
                             noise: float = 0.03, seed: int = 0):
    """
    Simple anomaly score stream for experiments (no dependency on ROS).
    """
    rng = np.random.default_rng(seed)
    s = base + noise * rng.standard_normal(T)
    s = np.clip(s, 0.0, 1.0)
    if 0 <= attack_start < attack_end <= T:
        s[attack_start:attack_end] = np.clip(
            attack_level + noise * rng.standard_normal(attack_end - attack_start),
            0.0, 1.0
        )
    return s
