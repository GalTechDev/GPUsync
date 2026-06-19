import os
import sys
import time
import asyncio
import threading
import subprocess
import csv
import psutil
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
except ImportError:
    print("Erreur: scikit-learn n'est pas installé.")
    sys.exit(1)

# Add paths to import GPUsync and kmeans_gpu
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'kmeans'))

from kmeans_gpu import kmeans_out_of_core

def get_vram_usage():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        return float(result.strip().split('\n')[0]) # en MB
    except Exception:
        return 0.0

class ResourceMonitor:
    def __init__(self):
        self.keep_running = False
        self.max_ram = 0.0
        self.max_vram = 0.0
        self.process = psutil.Process(os.getpid())
        
    def monitor(self):
        while self.keep_running:
            ram = self.process.memory_info().rss / (1024 * 1024) # en MB
            vram = get_vram_usage()
            if ram > self.max_ram:
                self.max_ram = ram
            if vram > self.max_vram:
                self.max_vram = vram
            time.sleep(0.05)
            
    def start(self):
        self.max_ram = 0.0
        self.max_vram = 0.0
        self.keep_running = True
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()
        
    def stop(self):
        self.keep_running = False
        self.thread.join()
        return self.max_ram, self.max_vram

async def main():
    print("==========================================================")
    print("--- BENCHMARK EXHAUSTIF K-MEANS : OUT-OF-CORE vs CPU ---")
    print("==========================================================")
    
    D = 64
    K = 16
    
    # 64 GB of RAM allows up to ~50M points easily
    sizes_to_test = [
        1_000_000,
        5_000_000,
        10_000_000,
        20_000_000,
        50_000_000
    ]
    
    results = []
    monitor = ResourceMonitor()
    
    for N in sizes_to_test:
        print(f"\n--- Test Charge: {N:,} points ({D} dims, {K} clusters) ---")
        
        print("-> Génération des données en cours...")
        # On regénère les données à chaque fois pour nettoyer la RAM entre les tests
        X, _ = make_blobs(n_samples=N, n_features=D, centers=K, random_state=42)
        X = X.astype(np.float32)
        
        # --- SCIKIT-LEARN ---
        print("-> Lancement Scikit-Learn (CPU In-Core)...")
        monitor.start()
        start_time = time.time()
        
        # We limit max_iter to 10 for both to have a fair time comparison
        kmeans_cpu = KMeans(n_clusters=K, init='random', n_init=1, max_iter=10, random_state=42)
        kmeans_cpu.fit(X)
        
        cpu_time = time.time() - start_time
        cpu_ram, cpu_vram = monitor.stop()
        cpu_inertia = kmeans_cpu.inertia_
        
        print(f"   [CPU] Temps: {cpu_time:.2f}s | RAM: {cpu_ram:,.0f} MB | Inertie: {cpu_inertia:,.2f}")
        
        # Extract initial centroids to use the exact same initialization for GPUsync
        init_centroids = kmeans_cpu.cluster_centers_
        
        # --- GPUSYNC ---
        print("-> Lancement GPUsync (GPU Out-Of-Core)...")
        monitor.start()
        start_time = time.time()
        
        gpu_centroids, gpu_compute_time = await kmeans_out_of_core(X, K, max_iters=10, initial_C=init_centroids)
        
        # Le chronomètre s'arrête ICI, car l'algo K-Means est terminé.
        gpu_time = time.time() - start_time
        gpu_ram, gpu_vram = monitor.stop()
        
        # Calcul de l'inertie GPU par paquets pour ne pas exploser la RAM
        print("   [GPU] Fin du calcul. Calcul manuel de l'inertie en cours (CPU seul)...")
        gpu_inertia = 0.0
        chunk_size = 1_000_000
        for i in range(0, X.shape[0], chunk_size):
            X_chunk = X[i:i+chunk_size]
            diff = X_chunk[:, np.newaxis, :] - gpu_centroids[np.newaxis, :, :]
            gpu_inertia += float(np.sum(np.min(np.sum(diff**2, axis=2), axis=1)))
        
        print(f"   [GPU] Temps: {gpu_time:.2f}s | RAM: {gpu_ram:,.0f} MB | VRAM Peak: {gpu_vram:,.0f} MB | Inertie: {gpu_inertia:,.2f}")
        
        # Différence d'erreur
        error_diff = abs(cpu_inertia - gpu_inertia) / cpu_inertia * 100.0
        
        results.append({
            "Taille": N,
            "CPU_Time": cpu_time,
            "GPU_Time": gpu_time,
            "GPU_Compute_Time": gpu_compute_time,
            "CPU_RAM": cpu_ram,
            "GPU_RAM": gpu_ram,
            "CPU_VRAM": cpu_vram,
            "GPU_VRAM": gpu_vram,
            "Error_Pct": error_diff
        })
        
        # Free memory explicitly
        del X
        del kmeans_cpu
        import gc
        gc.collect()
        
    # --- SAUVEGARDE ET GRAPHIQUE ---
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)
        
    csv_path = os.path.join(benchmark_dir, "results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n[SUCCES] Résultats sauvegardés dans {csv_path}")
    
    # Trace des 4 graphiques
    x_labels = [f"{r['Taille'] // 1_000_000}M" for r in results]
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance Exhaustive : GPUsync (Out-Of-Core) vs Scikit-Learn (In-Core)', fontsize=16, fontweight='bold')
    
    # 1. Temps d'Exécution
    axs[0, 0].plot(x_labels, [r['CPU_Time'] for r in results], marker='o', label='Scikit-Learn (CPU In-Core)', color='#e74c3c', linewidth=2)
    axs[0, 0].plot(x_labels, [r['GPU_Time'] for r in results], marker='s', label='GPUsync (Total + Transposition Numpy)', color='#f39c12', linewidth=2, linestyle='--')
    axs[0, 0].plot(x_labels, [r['GPU_Compute_Time'] for r in results], marker='^', label='GPUsync (VRAM + Hardware Kernel)', color='#2ecc71', linewidth=3)
    axs[0, 0].set_title("Temps d'Exécution (Secondes)", fontweight='bold')
    axs[0, 0].set_ylabel("Secondes (Moins est mieux)")
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    axs[0, 0].legend()
    
    # 2. Consommation RAM
    axs[0, 1].plot(x_labels, [r['CPU_RAM'] / 1024 for r in results], marker='o', label='Scikit-Learn (CPU)', color='#e74c3c', linewidth=2)
    axs[0, 1].plot(x_labels, [r['GPU_RAM'] / 1024 for r in results], marker='o', label='GPUsync (GPU)', color='#2ecc71', linewidth=2)
    axs[0, 1].set_title("Pic de Consommation RAM (Go)", fontweight='bold')
    axs[0, 1].set_ylabel("Gigaoctets (Go)")
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].legend()
    
    # 3. Consommation VRAM
    # CPU usage should theoretically be flat for VRAM, but GPU should show flat at chunk size ~50MB
    axs[1, 0].plot(x_labels, [r['CPU_VRAM'] for r in results], marker='o', label='Scikit-Learn (CPU)', color='#e74c3c', linewidth=2, linestyle=':')
    axs[1, 0].plot(x_labels, [r['GPU_VRAM'] for r in results], marker='o', label='GPUsync (GPU)', color='#3498db', linewidth=2)
    axs[1, 0].set_title("Pic de Consommation VRAM (Mo)", fontweight='bold')
    axs[1, 0].set_ylabel("Mégaoctets (Mo)")
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].legend()
    
    # 4. Différence d'Erreur (Inertie)
    axs[1, 1].bar(x_labels, [r['Error_Pct'] for r in results], color='#9b59b6')
    axs[1, 1].set_title("Écart de Précision Mathématique (%)", fontweight='bold')
    axs[1, 1].set_ylabel("Erreur Relative (%)")
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(benchmark_dir, "kmeans_metrics.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[SUCCES] Graphique sauvegardé dans {plot_path}")

if __name__ == "__main__":
    asyncio.run(main())
