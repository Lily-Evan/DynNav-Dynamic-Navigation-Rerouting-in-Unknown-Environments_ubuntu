import pandas as pd
import matplotlib.pyplot as plt

# ================================
# Load data
# ================================
candidates_csv = "multiobj_candidates.csv"
pareto_csv = "multiobj_pareto_front.csv"

print("[VIZ] Loading:", candidates_csv)
cand = pd.read_csv(candidates_csv)

print("[VIZ] Loading:", pareto_csv)
pareto = pd.read_csv(pareto_csv)

# ================================
# 1. ENTROPY GAIN vs DISTANCE
# ================================
plt.figure(figsize=(7, 6))
plt.scatter(
    cand["distance"], cand["entropy_gain"],
    c=cand["score"], cmap="viridis", s=40, label="Candidates"
)

# highlight Pareto front points
plt.scatter(
    pareto["distance"], pareto["entropy_gain"],
    marker="*", s=200, color="red", edgecolor="black",
    label="Pareto front"
)

plt.colorbar(label="Score (weighted)")
plt.xlabel("Distance")
plt.ylabel("Entropy Gain")
plt.title("Multi-objective: Distance vs Entropy Gain")
plt.legend()
plt.tight_layout()
plt.savefig("multiobj_entropy_viz.png", dpi=200)
print("[VIZ] Saved multiobj_entropy_viz.png")

# ================================
# 2. UNCERTAINTY GAIN vs DISTANCE
# ================================
plt.figure(figsize=(7, 6))
plt.scatter(
    cand["distance"], cand["uncertainty_gain"],
    c=cand["score"], cmap="viridis", s=40, label="Candidates"
)

plt.scatter(
    pareto["distance"], pareto["uncertainty_gain"],
    marker="*", s=200, color="red", edgecolor="black",
    label="Pareto front"
)

plt.colorbar(label="Score (weighted)")
plt.xlabel("Distance")
plt.ylabel("Uncertainty Gain")
plt.title("Multi-objective: Distance vs Uncertainty Gain")
plt.legend()
plt.tight_layout()
plt.savefig("multiobj_uncertainty_viz.png", dpi=200)
print("[VIZ] Saved multiobj_uncertainty_viz.png")

print("\n[VIZ] Completed successfully.")

