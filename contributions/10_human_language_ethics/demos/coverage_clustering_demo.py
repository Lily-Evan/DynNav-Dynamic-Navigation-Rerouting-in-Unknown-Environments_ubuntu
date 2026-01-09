import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# -------------------------------
# 1. Δημιουργία ψεύτικου coverage grid
# -------------------------------
# 0 = uncovered, 1 = covered
H, W = 80, 80
grid = np.ones((H, W), dtype=int)

# Φτιάχνουμε μερικές "τρύπες" (uncovered περιοχές) σε διαφορετικά σημεία
grid[10:20, 10:20] = 0      # περιοχή 1
grid[40:55, 5:18] = 0       # περιοχή 2
grid[50:70, 40:60] = 0      # περιοχή 3
grid[5:15, 50:60] = 0       # περιοχή 4 (λίγο πιο sparse)

# -------------------------------
# 2. Παίρνουμε τα coords των uncovered cells
# -------------------------------
uncovered_indices = np.argwhere(grid == 0)  # N x 2 πίνακας (row, col)

if uncovered_indices.shape[0] == 0:
    print("Δεν υπάρχουν uncovered cells στο grid!")
    exit(0)

# Μετατρέπουμε σε (x, y) με x=col, y=row για plotting
points = np.zeros_like(uncovered_indices, dtype=float)
points[:, 0] = uncovered_indices[:, 1]  # x
points[:, 1] = uncovered_indices[:, 0]  # y

# -------------------------------
# 3. DBSCAN clustering
# -------------------------------
# eps: μέγιστη απόσταση γειτονιάς
# min_samples: ελάχιστα σημεία για να θεωρηθεί cluster
eps = 2.0
min_samples = 10

db = DBSCAN(eps=eps, min_samples=min_samples)
labels = db.fit_predict(points)

# labels = -1 για θόρυβο (noise points)
unique_labels = np.unique(labels)
num_clusters = np.sum(unique_labels != -1)

print(f"Βρέθηκαν {num_clusters} clusters (DBSCAN), συνολικά points: {len(points)}")

# -------------------------------
# 4. Υπολογισμός κέντρων clusters (centroids)
# -------------------------------
cluster_centers = []
for lab in unique_labels:
    if lab == -1:
        continue
    cluster_points = points[labels == lab]
    cx = np.mean(cluster_points[:, 0])
    cy = np.mean(cluster_points[:, 1])
    cluster_centers.append((cx, cy))

# -------------------------------
# 5. Plot
# -------------------------------
plt.figure(figsize=(6, 6))

# Plot των uncovered points ανά cluster
for lab in unique_labels:
    mask = (labels == lab)
    pts = points[mask]
    if lab == -1:
        # noise points
        plt.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.3, marker='x', label='Noise' if lab == -1 else None)
    else:
        plt.scatter(pts[:, 0], pts[:, 1], s=8, label=f'Cluster {lab}')

# Plot των κέντρων
if cluster_centers:
    centers_arr = np.array(cluster_centers)
    plt.scatter(centers_arr[:, 0], centers_arr[:, 1], s=80, marker='*')

plt.gca().invert_yaxis()  # για να ταιριάζει με row/col αναπαράσταση
plt.xlabel("x (column)")
plt.ylabel("y (row)")
plt.title(f"Uncovered Cells Clustering with DBSCAN (clusters = {num_clusters})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("coverage_clusters.png", dpi=200)
plt.show()
