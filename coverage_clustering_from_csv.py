import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# ---------------------------------------------------
# 1. Διαβάζουμε το coverage CSV ως pandas DataFrame
# ---------------------------------------------------

csv_path = "coverage_grid.csv"
print(f"Φορτώνω CSV από: {csv_path}")

df = pd.read_csv(csv_path)

print("Στήλες CSV:", df.columns.tolist())

# Προσπάθεια να καταλάβουμε ποιες στήλες υπάρχουν
# ΣΥΝΗΘΕΣ ΜΟΡΦΗ στα projects: cell_id, x, y, covered
# Αν υπάρχει διαφορετική ονομασία, θα το δούμε.
# ---------------------------------------------------

possible_x = ["x", "col", "column", "grid_x"]
possible_y = ["y", "row", "grid_y"]
possible_cov = ["covered", "cover", "is_covered", "value", "cov"]

# βρίσκουμε την πραγματική στήλη x
for c in possible_x:
    if c in df.columns:
        col_x = c
        break
else:
    raise ValueError("❌ Δεν βρέθηκε στήλη x στο CSV! Δώσε screenshot των πρώτων γραμμών.")

# βρίσκουμε την πραγματική στήλη y
for c in possible_y:
    if c in df.columns:
        col_y = c
        break
else:
    raise ValueError("❌ Δεν βρέθηκε στήλη y στο CSV!")

# βρίσκουμε στήλη που δηλώνει αν είναι covered ή όχι
for c in possible_cov:
    if c in df.columns:
        col_cov = c
        break
else:
    raise ValueError("❌ Δεν βρέθηκε στήλη για covered/uncovered στο CSV!")

print(f"Χρησιμοποιώ στήλες: x={col_x}, y={col_y}, covered={col_cov}")

# ---------------------------------------------------
# 2. Φιλτράρουμε τις uncovered θέσεις
# ---------------------------------------------------

# Αν covered = 1 σημαίνει covered, 0 = uncovered
df_uncovered = df[df[col_cov] == 0]

if len(df_uncovered) == 0:
    print("Δεν υπάρχουν uncovered cells στο CSV!")
    exit(0)

print(f"Uncovered cells: {len(df_uncovered)}")

# Παίρνουμε τα σημεία (x, y)
points = df_uncovered[[col_x, col_y]].to_numpy()

# ---------------------------------------------------
# 3. DBSCAN Clustering
# ---------------------------------------------------

eps = 2.0
min_samples = 10

db = DBSCAN(eps=eps, min_samples=min_samples)
labels = db.fit_predict(points)

unique_labels = np.unique(labels)
num_clusters = np.sum(unique_labels != -1)

print(f"Βρέθηκαν {num_clusters} clusters.")

# ---------------------------------------------------
# 4. Υπολογισμός κέντρων clusters
# ---------------------------------------------------

cluster_centers = []
for lab in unique_labels:
    if lab == -1:
        continue
    pts = points[labels == lab]
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    cluster_centers.append((cx, cy))

print("Κέντρα clusters:")
for i, c in enumerate(cluster_centers):
    print(f"  Cluster {i}: {c}")

# ---------------------------------------------------
# 5. Αποθήκευση waypoints
# ---------------------------------------------------

if len(cluster_centers) > 0:
    np.savetxt("cluster_waypoints.csv",
               np.array(cluster_centers),
               delimiter=",",
               header="x,y", comments="")
    print("Αποθηκεύτηκαν στο: cluster_waypoints.csv")

# ---------------------------------------------------
# 6. Plot
# ---------------------------------------------------

plt.figure(figsize=(6, 6))

for lab in unique_labels:
    pts = points[labels == lab]
    if lab == -1:
        plt.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.3, marker='x', label="Noise")
    else:
        plt.scatter(pts[:, 0], pts[:, 1], s=10, label=f"Cluster {lab}")

# Plot των centroids
if cluster_centers:
    centers = np.array(cluster_centers)
    plt.scatter(centers[:, 0], centers[:, 1], s=100, marker='*', color='black')

plt.title(f"DBSCAN Clusters (found {num_clusters})")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig("coverage_clusters_from_csv.png", dpi=200)
plt.show()

