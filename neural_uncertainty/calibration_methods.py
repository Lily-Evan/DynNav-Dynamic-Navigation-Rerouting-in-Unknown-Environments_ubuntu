# neural_uncertainty/calibration_methods.py

import numpy as np
from dataclasses import dataclass
from typing import Optional
import pickle


@dataclass
class CalibrationMetrics:
    """
    Βασικά metrics calibration για να συγκρίνεις
    uncalibrated vs calibrated αβεβαιότητες.
    """
    ece: float
    mae: float
    mse: float
    nll: Optional[float] = None


def _safe_clip_sigma(sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Αποφεύγουμε αρνητικές ή μηδενικές τιμές σ."""
    return np.clip(sigma, eps, None)


def compute_calibration_metrics(
    sigma_pred: np.ndarray,
    error_true: np.ndarray,
    num_bins: int = 15,
) -> CalibrationMetrics:
    """
    Υπολογίζει ECE, MAE, MSE, NLL για ένα σετ από (σ_pred, |e|).

    sigma_pred: predicted uncertainty (π.χ. drift σ)
    error_true: πραγματικό σφάλμα (π.χ. |drift| σε m ή deg)
    """
    sigma_pred = np.asarray(sigma_pred, dtype=float)
    error_true = np.asarray(error_true, dtype=float)

    mae = float(np.mean(np.abs(sigma_pred - error_true)))
    mse = float(np.mean((sigma_pred - error_true) ** 2))

    # Quantile-based bins
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(sigma_pred, quantiles)

    # Λίγο expand για να πιάσουμε όλα τα samples
    bin_edges[0] -= 1e-8
    bin_edges[-1] += 1e-8

    ece = 0.0
    n_total = len(sigma_pred)

    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (sigma_pred > low) & (sigma_pred <= high)
        if not np.any(mask):
            continue

        bin_sigma_mean = float(np.mean(sigma_pred[mask]))
        bin_error_mean = float(np.mean(error_true[mask]))
        bin_weight = float(np.sum(mask)) / n_total

        ece += bin_weight * abs(bin_sigma_mean - bin_error_mean)

    # NLL αν το sigma_pred είναι std dev Gaussian με mean=0
    sigma_clipped = _safe_clip_sigma(sigma_pred)
    nll = 0.5 * np.mean(
        np.log(2.0 * np.pi * sigma_clipped ** 2)
        + (error_true ** 2) / (sigma_clipped ** 2)
    )

    return CalibrationMetrics(ece=ece, mae=mae, mse=mse, nll=float(nll))


class BaseCalibrator:
    """
    Βασική abstract κλάση για calibrators:
      - fit(sigma_pred, error_true)
      - predict(sigma_pred)
    """

    def fit(self, sigma_pred: np.ndarray, error_true: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, sigma_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)


class AffineCalibrator(BaseCalibrator):
    """
    Affine calibration:
        sigma_calib = a * sigma_pred + b

    Τα (a, b) τα μαθαίνουμε με least squares πάνω στο |error_true|.
    """

    def __init__(self):
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, sigma_pred: np.ndarray, error_true: np.ndarray) -> None:
        sigma_pred = np.asarray(sigma_pred, dtype=float)
        error_true = np.asarray(error_true, dtype=float)

        x = sigma_pred
        y = error_true

        X = np.vstack([x, np.ones_like(x)]).T  # (N, 2)
        theta, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.a, self.b = float(theta[0]), float(theta[1])

    def predict(self, sigma_pred: np.ndarray) -> np.ndarray:
        sigma_pred = np.asarray(sigma_pred, dtype=float)
        sigma_calib = self.a * sigma_pred + self.b
        return _safe_clip_sigma(sigma_calib)


class HistogramCalibrator(BaseCalibrator):
    """
    Histogram / bin-based calibration:
      - group by bins σε επίπεδα sigma_pred
      - σε κάθε bin αποθηκεύουμε mean(error_true)
      - predict: αντικαθιστούμε sigma_pred με το mean του bin.

    Non-parametric μέθοδος calibration.
    """

    def __init__(self, num_bins: int = 15):
        self.num_bins = num_bins
        self.bin_edges: Optional[np.ndarray] = None
        self.bin_values: Optional[np.ndarray] = None

    def fit(self, sigma_pred: np.ndarray, error_true: np.ndarray) -> None:
        sigma_pred = np.asarray(sigma_pred, dtype=float)
        error_true = np.asarray(error_true, dtype=float)

        quantiles = np.linspace(0.0, 1.0, self.num_bins + 1)
        self.bin_edges = np.quantile(sigma_pred, quantiles)

        self.bin_edges[0] -= 1e-8
        self.bin_edges[-1] += 1e-8

        bin_values = []
        for i in range(self.num_bins):
            low, high = self.bin_edges[i], self.bin_edges[i + 1]
            mask = (sigma_pred > low) & (sigma_pred <= high)
            if not np.any(mask):
                bin_values.append(0.0)
            else:
                bin_values.append(float(np.mean(error_true[mask])))

        self.bin_values = np.array(bin_values, dtype=float)

    def predict(self, sigma_pred: np.ndarray) -> np.ndarray:
        if self.bin_edges is None or self.bin_values is None:
            raise RuntimeError("HistogramCalibrator must be fitted before predict().")

        sigma_pred = np.asarray(sigma_pred, dtype=float)
        sigma_calib = np.zeros_like(sigma_pred)

        indices = np.digitize(sigma_pred, self.bin_edges[1:-1], right=True)

        for idx in range(self.num_bins):
            mask = indices == idx
            if not np.any(mask):
                continue

            value = self.bin_values[idx]
            if value <= 0.0:
                sigma_calib[mask] = sigma_pred[mask]
            else:
                sigma_calib[mask] = value

        return _safe_clip_sigma(sigma_calib)
