import time
import asyncio
import numpy as np
import sys
import os
import psutil
import csv
import importlib.util

try:
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
except ImportError:
    print("Erreur: scikit-learn n'est pas installe. Lancez 'pip install scikit-learn'.")
    sys.exit(1)

# ─── Chargement des modules GPU ───────────────────────────────────────────────

def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_v1 = _load_module("kmeans_gpu_v1_mod", "kmeans_gpu_v1.py")
_v2 = _load_module("kmeans_gpu_v2_mod", "kmeans_gpu_v2.py")

kmeans_v1 = _v1.kmeans_out_of_core
# Fallback flexible pour accepter kmeans_out_of_core_v2 ou v2_5 selon ton fichier
kmeans_v2 = getattr(_v2, "kmeans_out_of_core_v2_5", getattr(_v2, "kmeans_out_of_core_v2", None))

if kmeans_v2 == None:
    print("Erreur : Impossible de trouver la fonction cible dans kmeans_gpu_v2.py")
    sys.exit(1)

# ─── RAM Monitor ─────────────────────────────────────────────────────────────

async def monitor_ram(stop_event):
    process = psutil.Process(os.getpid())
    peak_ram = 0.0
    while not stop_event.is_set():
        rss_mb = process.memory_info().rss / (1024 * 1024)
        if rss_mb > peak_ram:
            peak_ram = rss_mb
        await asyncio.sleep(0.01)
    return peak_ram

# ─── CPU Baseline ─────────────────────────────────────────────────────────────

def cpu_sync(X, K, initial_C):
    start = time.perf_counter()
    km = KMeans(n_clusters=K, init=initial_C, n_init=1, max_iter=20, algorithm='lloyd', tol=0.0)
    km.fit(X)
    elapsed = time.perf_counter() - start
    
    # On calcule le débit en considérant qu'on a fait 'km.n_iter_' passes complètes sur les données
    total_points_processed = X.shape[0] * km.n_iter_
    return elapsed, total_points_processed / elapsed, km.cluster_centers_

# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    sizes_to_test = [100_000, 250_000, 500_000, 1_000_000, 2_000_000]
    D = 64
    K = 16

    benchmark_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark", "kmeans"
    )
    os.makedirs(benchmark_dir, exist_ok=True)
    csv_path = os.path.join(benchmark_dir, "benchmark_results.csv")

    kernels = [
        {"name": "Scikit-Learn (CPU)", "type": "cpu"},
        {"name": "GalTech V1 (GPU)",   "type": "v1"},
        {"name": "GalTech V2 (GPU)",   "type": "v2"},
    ]
    colors = ['#ff6b6b', '#ffa500', '#00a8ff']

    all_results = {k["name"]: [] for k in kernels}
    existing_runs = set()

    # ─── REPRISE DU CSV EXISTANT (Évite les doublons) ─────────────────────────
    if os.path.exists(csv_path):
        print(f"[INFO] Analyse du fichier de résultats existant : {csv_path}")
        try:
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Sauter l'en-tête
                if header:
                    for row in reader:
                        if len(row) >= 5:
                            k_name, pts, t_sec, d_sec, ram = row[0], int(row[1]), float(row[2]), float(row[3]), float(row[4])
                            existing_runs.add((k_name, pts))
                            if k_name in all_results:
                                all_results[k_name].append({
                                    "Points": pts, "Temps_sec": t_sec, "Debit_sec": d_sec, "RAM_MB": ram
                                })
            print(f"[OK] {len(existing_runs)} benchmarks chargés depuis l'historique.")
        except Exception as e:
            print(f"[ATTENTION] Erreur lors de la lecture du CSV existant : {e}. Il sera réinitialisé.")
            existing_runs = set()

    # Si le fichier n'existe pas, on l'initialise avec son en-tête
    if not os.path.exists(csv_path) or not existing_runs:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["Kernel", "Points", "Temps_sec", "Debit_sec", "RAM_MB"])

    # ─── IMAGE 2 : Validation visuelle du clustering 2D ──────────────────────
    print("\n[INFO] Generation du test visuel 2D (3 colonnes : CPU / V1 / V2)...")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        N_vis, D_vis, K_vis = 10_000, 2, 5
        X_vis, _ = make_blobs(n_samples=N_vis, n_features=D_vis, centers=K_vis, random_state=42)
        X_vis = X_vis.astype(np.float32)
        np.random.seed(42)
        init_C_vis = X_vis[np.random.choice(N_vis, K_vis, replace=False)].copy()

        # CPU
        km_cpu = KMeans(n_clusters=K_vis, init=init_C_vis, n_init=1, max_iter=20, tol=0.0, algorithm='lloyd')
        km_cpu.fit(X_vis)
        cpu_centers = km_cpu.cluster_centers_
        cpu_labels  = km_cpu.labels_

        # V1
        v1_centers, _ = await kmeans_v1(X_vis, K_vis, initial_C=init_C_vis, max_iters=20, tol=0.0, backend="opencl")
        v1_labels = np.argmin(np.linalg.norm(X_vis[:, None] - v1_centers, axis=2), axis=1)

        # V2
        v2_centers, _ = await kmeans_v2(X_vis, K_vis, initial_C=init_C_vis, max_iters=20, tol=0.0, backend="opencl")
        v2_labels = np.argmin(np.linalg.norm(X_vis[:, None] - v2_centers, axis=2), axis=1)

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle("Validation visuelle du clustering K-Means (N=10 000, D=2, K=5)", fontsize=14, fontweight='bold')

        for ax, labels, centers, title in [
            (axes[0], cpu_labels, cpu_centers, "Scikit-Learn (CPU)"),
            (axes[1], v1_labels,  v1_centers,  "GalTech V1 (GPU)"),
            (axes[2], v2_labels,  v2_centers,  "GalTech V2 (GPU)"),
        ]:
            ax.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, s=2, cmap='tab10', alpha=0.5)
            ax.scatter(centers[:, 0], centers[:, 1], c='white', s=120, marker='X',
                       edgecolors='black', linewidths=1.2, zorder=5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        vis_path = os.path.join(benchmark_dir, "clustering_visual_comparison.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Image visuelle : {vis_path}")
    except Exception as e:
        import traceback
        print(f"[ERREUR] Image visuelle: {e}")
        traceback.print_exc()

    # ─── BENCHMARK DE PERFORMANCE ─────────────────────────────────────────────
    print("\n[INFO] Demarrage du Benchmark de Performance...")

    for s in sizes_to_test:
        # Vérification si TOUS les kernels pour cette taille existent déjà
        all_kernels_done = all((k["name"], s) in existing_runs for k in kernels)
        if all_kernels_done:
            print(f"\n[SKIP] Charge {s:,} Points dejà intégralement présente dans le CSV.")
            continue

        print(f"\n{'='*52}")
        print(f"Charge : {s:,} Points | {D} Dim | {K} Clusters")
        print(f"{'='*52}")

        X, _ = make_blobs(n_samples=s, n_features=D, centers=K, random_state=42)
        X = X.astype(np.float32)
        np.random.seed(42)
        initial_C = X[np.random.choice(s, K, replace=False)].copy()

        for kernel in kernels:
            name = kernel["name"]
            ktype = kernel["type"]

            # Tri sélectif des configurations exécutées
            if (name, s) in existing_runs:
                print(f"--- {name} : Déjà calculé (Saut de l'étape) ---")
                continue

            print(f"\n--- {name} ---")

            stop_event = asyncio.Event()
            monitor_task = asyncio.create_task(monitor_ram(stop_event))

            try:
                if ktype == "cpu":
                    loop = asyncio.get_running_loop()
                    elapsed, debit, _ = await loop.run_in_executor(None, cpu_sync, X, K, initial_C)
                elif ktype == "v1":
                    _, elapsed = await kmeans_v1(X, K, initial_C=initial_C, max_iters=20, tol=0.0, backend="opencl")
                    debit = (s * 20) / elapsed
                else:  # v2 / v2.5
                    _, elapsed = await kmeans_v2(X, K, initial_C=initial_C, max_iters=20, tol=0.0, backend="opencl")
                    debit = (s * 20) / elapsed
            except Exception as e:
                print(f"  [ERREUR] {e}")
                stop_event.set()
                await monitor_task
                continue

            stop_event.set()
            peak_ram = await monitor_task

            print(f"-> {elapsed:.2f}s | {debit:,.0f} pts/s | RAM: {peak_ram:.1f} Mo")

            all_results[name].append({
                "Points": s, "Temps_sec": elapsed, "Debit_sec": debit, "RAM_MB": peak_ram
            })
            with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([name, s, elapsed, debit, peak_ram])

    # ─── IMAGE 1 : Graphique de performance (tous sur le meme plot) ───────────
    print("\n[INFO] Generation du graphique de performance...")
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("K-Means : Comparaison GPU V1 / GPU V2 / CPU", fontsize=15, fontweight='bold')

        def format_tick(v):
            return f"{v/1_000_000:.0f}M" if v >= 1_000_000 else f"{v/1_000:.0f}k"

        for idx, kernel in enumerate(kernels):
            name    = kernel["name"]
            results = all_results[name]
            if not results:
                continue

            # Tri impératif par nombre de points pour éviter les lignes croisées (en cas d'execution partielle)
            results = sorted(results, key=lambda x: x["Points"])

            x_vals  = [r['Points']   for r in results]
            y_debit = [r['Debit_sec'] / 1_000 for r in results]
            y_ram   = [r['RAM_MB']   for r in results]
            c       = colors[idx]

            ax1.plot(x_vals, y_debit, marker='o', linestyle='-', color=c,
                     linewidth=2.5, markersize=7, label=name)
            for xi, yi in zip(x_vals, y_debit):
                ax1.annotate(f"{yi:.0f}k", (xi, yi), textcoords="offset points",
                             xytext=(0, 8), ha='center', fontsize=7, color=c, weight='bold')

            ax2.plot(x_vals, y_ram, marker='s', linestyle='--', color=c,
                     linewidth=2, markersize=6, label=name)
            for xi, yi in zip(x_vals, y_ram):
                ax2.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                             xytext=(0, -12), ha='center', fontsize=7, color=c)

        for ax in (ax1, ax2):
            ax.set_xscale('log')
            ax.grid(True, which='both', ls='--', alpha=0.3)
            ax.legend(loc='upper left', fontsize=10)

        plt.xticks(sizes_to_test, [format_tick(s) for s in sizes_to_test])
        ax1.set_ylabel("Debit (k pts/s)", fontsize=11)
        ax1.set_title("Debit de convergence", fontsize=12, pad=6)
        ax2.set_ylabel("RAM consommee (Mo)", fontsize=11)
        ax2.set_xlabel("Nombre de points", fontsize=11)
        ax2.set_title("Consommation RAM hote (RSS)", fontsize=12, pad=6)

        plt.tight_layout()
        perf_path = os.path.join(benchmark_dir, "throughput_comparison_plot.png")
        plt.savefig(perf_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Graphique perf : {perf_path}")

    except Exception as e:
        print(f"[ERREUR] Graphique perf : {e}")

if __name__ == "__main__":
    asyncio.run(main())