#!/usr/bin/env python3
import numpy as np
from sklearn.cluster import DBSCAN

def cluster_uncovered(uncovered_cells, eps=0.5, min_samples=3):
    """
    uncovered_cells: list of (x,y)
    returns: {cluster_id: [(x,y),...]}
    """

    if len(uncovered_cells) == 0:
        return {}

    X = np.array(uncovered_cells)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    clusters = {}
    for pt, c in zip(uncovered_cells, labels):
        if c == -1:
            continue
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(pt)

    return clusters

