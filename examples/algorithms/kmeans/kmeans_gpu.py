import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from gpusync import gpusync_kernel

@gpusync_kernel
def assign_clusters_kernel(ctx: int, X: list, centroids: list, n_clusters: int, dim: int) -> tuple:
    """
    Assigne chaque point X[ctx] au centroid le plus proche.
    """
    best_dist = 1e9
    best_cluster = 0
    
    for k in range(n_clusters):
        dist = 0.0
        for d in range(dim):
            diff = X[ctx * dim + d] - centroids[k * dim + d]
            dist = dist + diff * diff
            
        if dist < best_dist:
            best_dist = dist
            best_cluster = k
            
    return float(best_cluster)
