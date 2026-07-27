import sys
import os
import numpy as np

# Ajouter le chemin vers la racine du projet
sys.path.insert(0, os.path.abspath('.'))

from examples.algorithms.xmeans.xmeans_engine import XMeansEngine

print("Generating data...")
np.random.seed(42)
N = 10000
D = 8
X = np.random.rand(N, D).astype(np.float32)

engine = XMeansEngine(dim=D, max_k=16)

# Create scheduler
from gpusync import Scheduler, list_gpus
gpus = list_gpus()
sched = Scheduler(gpu=gpus[0])
sched.initialize()

print("Running regular XMeans...")
try:
    labels, centroids = engine.fit(sched, X)
    print(f"XMeans labels: {labels.shape}, centroids: {centroids.shape}")
except Exception as e:
    print(f"Error in regular XMeans: {e}")

print("Running Genetic XMeans...")
try:
    labels_gen, centroids_gen = engine.fit_genetic(sched, X, P=8)
    print(f"Genetic XMeans labels: {labels_gen.shape}, centroids: {centroids_gen.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error in Genetic XMeans: {e}")

print("Done.")
