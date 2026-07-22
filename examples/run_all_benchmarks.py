import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpusync import Scheduler, list_gpus
from examples.algorithms.dtw.benchmark import DTWBenchmark
from examples.algorithms.kmeans.benchmark import KMeansBenchmark
from examples.algorithms.matrix.benchmark import MatrixBenchmark
from examples.algorithms.knn_dtw.benchmark import KNNDTWBenchmark
from examples.algorithms.kshape.benchmark import KShapeBenchmark
from examples.algorithms.local_regression.benchmark import LOESSBenchmark
from examples.algorithms.matrix_profile.benchmark import STOMPBenchmark
from examples.algorithms.gr_soft_dtw.benchmark import GRSoftDTWBenchmark
from examples.algorithms.xmeans.benchmark import XMeansBenchmark

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("=========================================================================", flush=True)
    print("--- GPUSYNC MASTER BENCHMARK RUNNER ---", flush=True)
    print("=========================================================================", flush=True)

    gpus = list_gpus()
    if not gpus:
        print("[!] Aucun GPU détecté. Abandon du master benchmark.", flush=True)
        return

    gpu = gpus[0]
    print(f"[+] Périphérique cible : {gpu.name} ({gpu.backend})", flush=True)

    sched = Scheduler(gpu=gpu)
    sched.initialize()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Définition des 9 benchmarks
    benchmarks = [
        (DTWBenchmark(num_runs=2), [10_000, 100_000, 1_000_000, 2_000_000]),
        (KMeansBenchmark(num_runs=2), [50_000, 250_000]),
        (XMeansBenchmark(num_runs=2), [10_000, 50_000]),
        (MatrixBenchmark(num_runs=2), [128, 256, 512]),
        (KNNDTWBenchmark(num_runs=2), [10_000, 25_000]),
        (KShapeBenchmark(num_runs=2), [100, 250]),
        (LOESSBenchmark(num_runs=2), [10_000, 25_000]),
        (STOMPBenchmark(num_runs=2), [10_000, 25_000]),
        (GRSoftDTWBenchmark(num_runs=2), [50, 75])
    ]

    all_summaries = []

    for bm, sizes in benchmarks:
        bm.execute(sizes=sizes, scheduler=sched, results_dir=results_dir)
        all_summaries.append({
            "name": bm.name,
            "description": bm.description,
            "variants": bm.variants,
            "results": bm.results
        })

    reports_dir = os.path.join(results_dir, "reports")
    summary_md_path = os.path.join(reports_dir, "BENCHMARK_SUMMARY.md")

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# GPUsync Global Benchmark Summary Report\n\n")
        f.write(f"- **Périphérique de calcul** : {gpu.name} (`{gpu.backend}`)\n")
        f.write(f"- **Date de génération** : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for s in all_summaries:
            f.write(f"## {s['name']}\n")
            f.write(f"*{s['description']}*\n\n")
            f.write("| Taille N | Temps CPU (s) | Variante | Temps GPU (s) | Speedup |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for r in s['results']:
                for idx, var in enumerate(s['variants']):
                    v_res = r['variants'].get(var, {})
                    size_str = f"{r['size']:,}" if idx == 0 else ""
                    cpu_str = f"{r['mean_cpu']:.4f}s" if idx == 0 else ""
                    if v_res.get('skipped', False):
                        f.write(f"| {size_str} | {cpu_str} | {var} | Skipped | N/A |\n")
                    else:
                        f.write(f"| {size_str} | {cpu_str} | {var} | {v_res.get('mean', 0.0):.4f}s | **{v_res.get('speedup', 0.0):.2f}x** |\n")
            f.write("\n")

    print("\n" + "=" * 80, flush=True)
    print(f"[OK] MASTER BENCHMARK TERMINÉ AVEC SUCCÈS !", flush=True)
    print(f"[OK] Rapport Markdown disponible : {summary_md_path}", flush=True)
    print(f"[OK] Graphiques PNG disponibles  : {os.path.join(results_dir, 'plots')}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
