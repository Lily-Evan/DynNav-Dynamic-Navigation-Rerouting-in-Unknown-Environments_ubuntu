import os
import numpy as np
import matplotlib.pyplot as plt

# ================================
# Plots για Real-World Evaluation
# Αποθηκεύονται ως εικόνες (.png)
# ================================

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, "logs_real_world")

mse_before = np.load(os.path.join(log_dir, "mse_before.npy"))
mse_after = np.load(os.path.join(log_dir, "mse_after.npy"))
S = np.load(os.path.join(log_dir, "self_trust.npy"))

print("[PLOT] Loaded logs from:", log_dir)
print(f"[PLOT] Online samples = {len(mse_before)}")

# ==================== Plot 1: MSE ====================
plt.figure(figsize=(10,5))
plt.plot(mse_before, label="MSE Before Online Adaptation", color="red")
plt.plot(mse_after, label="MSE After Online Adaptation", color="green")
plt.xlabel("Online Sample Index")
plt.ylabel("Squared Error")
plt.title("Online Adaptation MSE Comparison (Real-World Style)")
plt.legend()
plt.grid()

mse_plot_path = os.path.join(log_dir, "real_world_mse_plot.png")
plt.savefig(mse_plot_path, dpi=200)
print(f"[PLOT] Saved: {mse_plot_path}")
plt.close()

# ==================== Plot 2: Self Trust ====================
plt.figure(figsize=(10,5))
plt.plot(S, label="Self-Trust Score S", color="blue")
plt.xlabel("Online Sample Index")
plt.ylabel("Self-Trust Score (0–1)")
plt.ylim(0,1)
plt.title("Online Self-Trust Evolution (Real-World Style)")
plt.grid()
plt.legend()

st_plot_path = os.path.join(log_dir, "real_world_self_trust_plot.png")
plt.savefig(st_plot_path, dpi=200)
print(f"[PLOT] Saved: {st_plot_path}")
plt.close()

print("\n[PLOT] All images saved successfully 🎯")
