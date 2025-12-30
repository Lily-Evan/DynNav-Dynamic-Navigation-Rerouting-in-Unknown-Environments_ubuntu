# ================================================================
# Ensemble-based Drift Uncertainty Adapter (Experimental)
#
# Χτίζει ένα ensemble από OnlineDriftUncertaintyAdapterExp instances.
# Το mean είναι ο μέσος όρος των μέσων,
# και η συνολική epistemic variance έχει όρο διασποράς μεταξύ των members.
# ================================================================

import numpy as np
import torch

from online_drift_uncertainty_exp import OnlineDriftUncertaintyAdapterExp


class EnsembleDriftUncertaintyAdapterExp:
    def __init__(
        self,
        n_members: int,
        input_dim: int,
        lr: float = 1e-4,
        device: str = "cpu",
        max_buffer_size: int = 512,
        weight_decay: float = 1e-6,
        dropout_p: float = 0.1,
        base_model_path=None,
        base_seed: int = 0,
    ):
        self.n_members = n_members
        self.device = device
        self.members = []

        for i in range(n_members):
            seed_i = base_seed + i
            np.random.seed(seed_i)
            torch.manual_seed(seed_i)

            adapter = OnlineDriftUncertaintyAdapterExp(
                model_path=base_model_path,
                input_dim=input_dim,
                lr=lr,
                device=device,
                max_buffer_size=max_buffer_size,
                weight_decay=weight_decay,
                dropout_p=dropout_p,
            )
            self.members.append(adapter)

    def add_observation(self, x, y):
        for m in self.members:
            m.add_observation(x, y)

    def online_update(self, batch_size: int = 32):
        losses = []
        for m in self.members:
            loss = m.online_update(batch_size=batch_size)
            losses.append(loss)
        return float(np.mean(losses))

    def predict_means_only(self, x):
        """
        Επιστρέφει ένα array με όλα τα per-member means.
        """
        x_arr = np.array(x, dtype=np.float32)
        means = []
        for m in self.members:
            pred = m.predict(x_arr).reshape(-1)[0]
            means.append(pred)
        return np.array(means, dtype=np.float32)

    def predict_with_uncertainty(self, x, n_mc: int = 0):
        """
        Επιστρέφει συνολικό mean & epistemic variance από το ensemble.
        Προαιρετικά μπορεί να χρησιμοποιεί και MC dropout per member (n_mc > 0).
        """
        x_arr = np.array(x, dtype=np.float32)

        if n_mc <= 1:
            # μόνο ensemble
            means = self.predict_means_only(x_arr)
            mean_global = float(means.mean())
            var_epistemic = float(means.var() + 1e-8)
            return np.array([mean_global], dtype=np.float32), np.array([var_epistemic], dtype=np.float32)

        # MC Dropout μέσα σε κάθε member + ensemble
        all_samples = []
        for m in self.members:
            samples = []
            for _ in range(n_mc):
                mu, _ = m.predict_with_uncertainty(x_arr, n_samples=1)
                samples.append(mu.reshape(-1)[0])
            all_samples.extend(samples)

        all_samples = np.array(all_samples, dtype=np.float32)
        mean_global = float(all_samples.mean())
        var_global = float(all_samples.var() + 1e-8)
        return np.array([mean_global], dtype=np.float32), np.array([var_global], dtype=np.float32)
