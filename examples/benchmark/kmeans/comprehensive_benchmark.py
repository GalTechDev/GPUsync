import os
import sys
import time
import asyncio
import numpy as np
import csv
import psutil
import gc
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Ajouter le chemin racine au path Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from examples.kmeans.kmeans_gpu import kmeans_out_of_core
import cupy as cp

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

def get_vram_usage():
    mempool = cp.get_default_memory_pool()
    return mempool.used_bytes() / 1e9

async def run_gpusync(X, K, initial_C, max_iters):
    start_vram = get_vram_usage()
    start_ram = get_ram_usage()
    
    start_time = time.perf_counter()
    C_gpu = await kmeans_out_of_core(X, K, initial_C=initial_C, max_iters=max_iters, backend="cuda")
    exec_time = time.perf_counter() - start_time
    
    peak_vram = get_vram_usage() - start_vram
    peak_ram = get_ram_usage() - start_ram
    return exec_time, max(0, peak_vram), max(0, peak_ram)

def run_sklearn(X, K, initial_C, max_iters):
    start_ram = get_ram_usage()
    
    start_time = time.perf_counter()
    kmeans = KMeans(n_clusters=K, init=initial_C, n_init=1, max_iter=max_iters, algorithm='lloyd')
    kmeans.fit(X)
    exec_time = time.perf_counter() - start_time
    
    peak_ram = get_ram_usage() - start_ram
    return exec_time, max(0, peak_ram)

async def main():
    print("==================================================")
    print("   COMPREHENSIVE BENCHMARK: K-MEANS OUT-OF-CORE   ")
    print("==================================================")
    
    results = []
    
    # Tailles de dataset: 100k, 1M, 10M, 100M
    sizes = [100_000, 1_000_000, 10_000_000, 100_000_000]
    D = 64
    K = 16
    max_iters = 10
    
    csv_file = os.path.join(os.path.dirname(__file__), 'results.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'Impl', 'Time_s', 'VRAM_GB', 'RAM_GB'])
    
    for N in sizes:
        print(f"\n>>> Démarrage Test N = {N:,} points (Dim={D})")
        
        print(f"Génération du dataset...")
        if N >= 10_000_000:
            print("  Génération via Memmap sur le disque pour éviter le crash de la RAM (Out-Of-Core pur)...")
            memmap_path = os.path.join(os.path.dirname(__file__), 'temp_dataset.dat')
            X = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(N, D))
            
            # Génération par blocs de 1M pour ne pas saturer la RAM
            chunk_gen = 1_000_000
            for i in range(0, N, chunk_gen):
                end_i = min(i + chunk_gen, N)
                X[i:end_i] = np.random.randn(end_i - i, D).astype(np.float32)
            X.flush()
        else:
            X, _ = make_blobs(n_samples=N, n_features=D, centers=K, random_state=42)
            X = X.astype(np.float32)
        
        np.random.seed(42)
        random_indices = np.random.choice(N, K, replace=False)
        initial_C = X[random_indices].copy().astype(np.float32)
        
        # Scikit-Learn
        if N <= 10_000_000:
            print("Lancement Scikit-Learn...")
            gc.collect()
            time_sk, ram_sk = run_sklearn(X, K, initial_C, max_iters)
            print(f"  Scikit-Learn: {time_sk:.2f}s | RAM Peak: {ram_sk:.2f}GB")
            
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([N, 'Scikit-Learn', time_sk, 0, ram_sk])
        else:
            print("Scikit-Learn ignoré (N trop grand pour éviter crash RAM)")
            
        # GPUsync
        print("Lancement GPUsync...")
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
        time_gpu, vram_gpu, ram_gpu = await run_gpusync(X, K, initial_C, max_iters)
        print(f"  GPUsync: {time_gpu:.2f}s | VRAM Peak: {vram_gpu:.2f}GB | RAM Peak: {ram_gpu:.2f}GB")
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([N, 'GPUsync', time_gpu, vram_gpu, ram_gpu])
            
        # Suppression explicite pour nettoyer la RAM pour la taille suivante
        del X
        del initial_C
        gc.collect()

if __name__ == "__main__":
    asyncio.run(main())
