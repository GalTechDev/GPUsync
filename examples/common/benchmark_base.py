import time
import os
import json
import numpy as np

class BaseBenchmark:
    """
    Classe de base réutilisable et standardisée pour tous les benchmarks GPUsync.
    Gère le calibrage, l'évaluation multi-runs, le calcul des speedups, l'affichage
    de tableaux ASCII et la génération automatique de plots PNG à la charte GPUsync.
    """
    def __init__(self, name: str, description: str, num_runs: int = 5):
        self.name = name
        self.description = description
        self.num_runs = num_runs
        self.results = []

    def setup_data(self, size: int):
        """Initialise les données pour une taille de problème donnée."""
        raise NotImplementedError

    def run_cpu(self, data):
        """Exécution de référence sur CPU (NumPy / SciPy / Sklearn)."""
        raise NotImplementedError

    def run_gpusync(self, data, scheduler):
        """Exécution optimisée sur GPUsync."""
        raise NotImplementedError

    def execute(self, sizes: list, scheduler, results_dir: str = "examples/results"):
        """
        Exécute la batterie de benchmarks sur les différentes tailles de problèmes.
        """
        import sys
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        print("\n" + "=" * 80)
        print(f"--- BENCHMARK : {self.name.upper()} ---")
        print(f"Description : {self.description}")
        print(f"Nombre de runs par taille : {self.num_runs}")
        print("=" * 80)

        self.results = []

        for size in sizes:
            print(f"\n[+] Évaluation pour la taille N = {size:,}...")
            data = self.setup_data(size)

            # Warmup
            try:
                _ = self.run_cpu(data)
                _ = self.run_gpusync(data, scheduler)
            except Exception as e:
                print(f"    [!] Erreur lors du Warmup : {e}")

            # 1. Évaluation CPU
            cpu_times = []
            for _ in range(self.num_runs):
                t0 = time.perf_counter()
                _ = self.run_cpu(data)
                t1 = time.perf_counter()
                cpu_times.append(t1 - t0)

            mean_cpu = float(np.mean(cpu_times))
            std_cpu = float(np.std(cpu_times))

            # 2. Évaluation GPUsync
            gpusync_times = []
            for _ in range(self.num_runs):
                t0 = time.perf_counter()
                _ = self.run_gpusync(data, scheduler)
                t1 = time.perf_counter()
                gpusync_times.append(t1 - t0)

            mean_gpusync = float(np.mean(gpusync_times))
            std_gpusync = float(np.std(gpusync_times))

            speedup = mean_cpu / max(mean_gpusync, 1e-7)

            item = {
                "size": size,
                "mean_cpu": mean_cpu,
                "std_cpu": std_cpu,
                "mean_gpusync": mean_gpusync,
                "std_gpusync": std_gpusync,
                "speedup": speedup
            }
            self.results.append(item)

            print(f"    -> CPU Mean     : {mean_cpu:.4f} s (±{std_cpu:.4f}s)")
            print(f"    -> GPUsync Mean : {mean_gpusync:.4f} s (±{std_gpusync:.4f}s)")
            print(f"    -> SPEEDUP      : {speedup:.2f}x")

        self.print_summary_table()
        self.export_results(results_dir)
        self.plot_results(results_dir)

    def print_summary_table(self):
        """Affiche un tableau ASCII formaté des résultats."""
        print("\n" + "-" * 80)
        print(f"--- RÉSULTATS SYNTHÉTIQUES : {self.name} ---")
        print("-" * 80)
        print(f"{'Taille N':<15} | {'Temps CPU (s)':<16} | {'GPUsync (s)':<16} | {'Speedup':<12}")
        print("-" * 80)
        for r in self.results:
            print(f"{r['size']:<15,} | {r['mean_cpu']:<16.4f} | {r['mean_gpusync']:<16.4f} | {r['speedup']:<12.2f}x")
        print("-" * 80)

    def export_results(self, results_dir: str):
        """Exporte les résultats au format JSON et Markdown."""
        reports_dir = os.path.join(results_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        json_path = os.path.join(reports_dir, f"{self.name.lower()}_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "name": self.name,
                "description": self.description,
                "num_runs": self.num_runs,
                "results": self.results
            }, f, indent=2)

        print(f"\n[OK] Résultats JSON sauvegardés dans : {json_path}")

    def plot_results(self, results_dir: str):
        """
        Génère un graphique PNG professionnel à la charte visuelle GPUsync
        (Fond sombre néon, axe double temps/speedup).
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[!] Matplotlib non disponible, saut de la génération de plot.")
            return

        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        sizes = [r['size'] for r in self.results]
        cpu_times = [r['mean_cpu'] for r in self.results]
        gpusync_times = [r['mean_gpusync'] for r in self.results]
        speedups = [r['speedup'] for r in self.results]

        # Style GPUsync Neon Dark
        plt.style.use('dark_background')
        fig, ax1 = plt.subplots(figsize=(10, 6), facecolor='#0B0F19')
        ax1.set_facecolor('#0B0F19')

        x_labels = [f"{s:,}" for s in sizes]
        x = np.arange(len(sizes))
        width = 0.35

        rects1 = ax1.bar(x - width/2, cpu_times, width, label='CPU (Single Thread)', color='#FF0055', alpha=0.85)
        rects2 = ax1.bar(x + width/2, gpusync_times, width, label='GPUsync (JIT/Hardware)', color='#00F0FF', alpha=0.85)

        ax1.set_xlabel('Taille du problème (N)', fontsize=12, color='#F8FAFC', labelpad=10)
        ax1.set_ylabel('Temps d\'exécution (Secondes)', fontsize=12, color='#F8FAFC', labelpad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, color='#CBD5E1')
        ax1.tick_params(colors='#CBD5E1')
        ax1.grid(True, linestyle='--', alpha=0.15, color='#94A3B8')

        # Axe 2 pour le Speedup
        ax2 = ax1.twinx()
        ax2.plot(x, speedups, color='#F59E0B', marker='o', linewidth=2.5, markersize=8, label='Speedup (x)')
        ax2.set_ylabel('Speedup vs CPU (x)', fontsize=12, color='#F59E0B', labelpad=10)
        ax2.tick_params(colors='#F59E0B')

        plt.title(f"GPUsync Benchmark — {self.name}", fontsize=14, color='#F8FAFC', pad=15, fontweight='bold')

        # Légende combinée
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor='#1E293B', edgecolor='#334155')

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"{self.name.lower()}_benchmark.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"[OK] Graphique PNG sauvegardé dans : {plot_path}")
