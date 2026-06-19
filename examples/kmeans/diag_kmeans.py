import sys
import os
import asyncio
import numpy as np
from sklearn.datasets import make_blobs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kmeans_gpu import kmeans_out_of_core

async def run_diagnostic():
    N = 100
    D = 4
    K = 3
    
    np.random.seed(42)
    X, _ = make_blobs(n_samples=N, n_features=D, centers=K, random_state=42)
    X = X.astype(np.float32)
    
    initial_C = X[:K].copy()
    
    print("=== Démarrage GPUsync ===")
    C_gpu = await kmeans_out_of_core(X, K, initial_C=initial_C, max_iters=1, backend="cuda")
    
    print("\n=== Pure Numpy (1 iteration) ===")
    dists = np.sum((X[:, np.newaxis, :] - initial_C[np.newaxis, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    
    C_np = np.zeros_like(initial_C)
    for k in range(K):
        pts = X[labels == k]
        if len(pts) > 0:
            C_np[k] = pts.mean(axis=0)
        else:
            C_np[k] = initial_C[k]
            
    print("\nCentroïdes Numpy:")
    print(C_np)
    print("\nCentroïdes GPU:")
    print(C_gpu)
    
    diff = np.max(np.abs(C_np - C_gpu))
    print(f"\nDifférence max: {diff}")

if __name__ == "__main__":
    asyncio.run(run_diagnostic())
