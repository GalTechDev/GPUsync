import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpusync import Scheduler, list_gpus
from examples.algorithms.dtw.benchmark import DTWBenchmark
from examples.algorithms.kmeans.benchmark import KMeansBenchmark
from examples.algorithms.matrix.benchmark import MatrixBenchmark

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

    bm1 = DTWBenchmark(num_runs=2)
    bm1.execute(sizes=[10_000, 50_000], scheduler=sched, results_dir=results_dir)

    bm2 = KMeansBenchmark(num_runs=2)
    bm2.execute(sizes=[50_000, 250_000], scheduler=sched, results_dir=results_dir)

    bm3 = MatrixBenchmark(num_runs=2)
    bm3.execute(sizes=[128, 256], scheduler=sched, results_dir=results_dir)

    all_summaries = [
        {"name": bm1.name, "description": bm1.description, "results": bm1.results},
        {"name": bm2.name, "description": bm2.description, "results": bm2.results},
        {"name": bm3.name, "description": bm3.description, "results": bm3.results}
    ]

    reports_dir = os.path.join(results_dir, "reports")
    summary_md_path = os.path.join(reports_dir, "BENCHMARK_SUMMARY.md")

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# GPUsync Global Benchmark Summary Report\n\n")
        f.write(f"- **Périphérique de calcul** : {gpu.name} (`{gpu.backend}`)\n")
        f.write(f"- **Date de génération** : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for s in all_summaries:
            f.write(f"## {s['name']}\n")
            f.write(f"*{s['description']}*\n\n")
            f.write("| Taille N | Temps CPU (s) | GPUsync (s) | Speedup |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for r in s['results']:
                f.write(f"| {r['size']:,} | {r['mean_cpu']:.4f}s | {r['mean_gpusync']:.4f}s | **{r['speedup']:.2f}x** |\n")
            f.write("\n")

    print("\n" + "=" * 80, flush=True)
    print(f"[OK] MASTER BENCHMARK TERMINÉ AVEC SUCCÈS !", flush=True)
    print(f"[OK] Rapport Markdown disponible : {summary_md_path}", flush=True)
    print(f"[OK] Graphiques PNG disponibles  : {os.path.join(results_dir, 'plots')}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
