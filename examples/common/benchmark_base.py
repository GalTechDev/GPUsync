import time
import os
import json
import numpy as np

class BaseBenchmark:
    """
    Classe de base réutilisable et standardisée pour tous les benchmarks GPUsync.
    Gère le calibrage, l'évaluation multi-runs de plusieurs variantes, le calcul des speedups,
    l'affichage de tableaux ASCII et la génération de graphiques PNG en barres groupées
    avec annotations de speedup et de temps sans axe Y double.
    """
    def __init__(self, name: str, description: str, num_runs: int = 5, variants: list = None):
        self.name = name
        self.description = description
        self.num_runs = num_runs
        self.variants = variants if variants is not None else ["GPUsync"]
        self.results = []

    def setup_data(self, size: int):
        """Initialise les données pour une taille de problème donnée."""
        raise NotImplementedError

    def run_cpu(self, data):
        """Exécution de référence sur CPU (NumPy / SciPy / Sklearn)."""
        raise NotImplementedError

    def run_gpusync(self, data, scheduler, variant: str):
        """Exécution optimisée sur GPUsync pour une variante donnée."""
        raise NotImplementedError

    def execute(self, sizes: list, scheduler, results_dir: str = "examples/results"):
        """
        Exécute la batterie de benchmarks sur les différentes tailles de problèmes et variantes.
        """
        import sys
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            
        print("\n" + "=" * 80)
        print(f"--- BENCHMARK : {self.name.upper()} ---")
        print(f"Description : {self.description}")
        print(f"Nombre de runs par taille : {self.num_runs}")
        print(f"Variantes évaluées : {', '.join(self.variants)}")
        print("=" * 80)

        self.results = []

        for size in sizes:
            print(f"\n[+] Évaluation pour la taille N = {size:,}...")
            data = self.setup_data(size)

            # 1. Évaluation CPU
            try:
                # Warmup CPU
                cpu_items = self.run_cpu(data)
                
                cpu_times = []
                for _ in range(self.num_runs):
                    t0 = time.perf_counter()
                    _ = self.run_cpu(data)
                    t1 = time.perf_counter()
                    cpu_times.append(t1 - t0)

                mean_cpu = float(np.mean(cpu_times))
                std_cpu = float(np.std(cpu_times))
                print(f"    -> CPU Mean     : {mean_cpu:.4f} s (±{std_cpu:.4f}s)")
            except Exception as e:
                print(f"    [!] Erreur lors de l'exécution CPU : {e}")
                mean_cpu = 0.0
                std_cpu = 0.0
                cpu_items = None

            # 2. Évaluation des variantes GPUsync
            variant_results = {}
            original_mean_cpu = mean_cpu
            original_std_cpu = std_cpu
            
            for var in self.variants:
                print(f"    -> Évaluation variante : {var}...")
                
                # Warmup & Support Check
                skipped = False
                try:
                    gpu_items = self.run_gpusync(data, scheduler, variant=var)
                except Exception as e:
                    print(f"        [!] Saut de la variante '{var}' (Non supportée ou erreur) : {e}")
                    skipped = True

                if skipped or original_mean_cpu == 0.0:
                    variant_results[var] = {
                        "mean": None,
                        "std": None,
                        "speedup": None,
                        "skipped": True
                    }
                    continue

                gpusync_times = []
                try:
                    for _ in range(self.num_runs):
                        t0 = time.perf_counter()
                        _ = self.run_gpusync(data, scheduler, variant=var)
                        t1 = time.perf_counter()
                        gpusync_times.append(t1 - t0)

                    mean_g = float(np.mean(gpusync_times))
                    std_g = float(np.std(gpusync_times))
                    
                    # Mise à l'échelle proportionnelle si la taille CPU est échantillonnée
                    if cpu_items and gpu_items and cpu_items != gpu_items:
                        scale = float(gpu_items) / float(cpu_items)
                        variant_mean_cpu = original_mean_cpu * scale
                        variant_std_cpu = original_std_cpu * scale
                        print(f"        [Scale] CPU mise à l'échelle par un facteur {scale:.2f}x (CPU items: {cpu_items:,} vs GPU items: {gpu_items:,})")
                    else:
                        variant_mean_cpu = original_mean_cpu
                        variant_std_cpu = original_std_cpu

                    speedup = variant_mean_cpu / max(mean_g, 1e-7)

                    variant_results[var] = {
                        "mean": mean_g,
                        "std": std_g,
                        "speedup": speedup,
                        "skipped": False,
                        "scaled_cpu_mean": variant_mean_cpu,
                        "scaled_cpu_std": variant_std_cpu
                    }
                    print(f"        -> Mean     : {mean_g:.4f} s (±{std_g:.4f}s)")
                    print(f"        -> SPEEDUP  : {speedup:.2f}x (vs CPU projeté {variant_mean_cpu:.4f}s)")
                except Exception as e:
                    print(f"        [!] Erreur lors de l'exécution de '{var}' : {e}")
                    variant_results[var] = {
                        "mean": None,
                        "std": None,
                        "speedup": None,
                        "skipped": True
                    }

            # Sélection du CPU mean final mis à l'échelle pour le rapport global
            final_cpu_mean = original_mean_cpu
            final_cpu_std = original_std_cpu
            for var in self.variants:
                v_res = variant_results.get(var, {})
                if not v_res.get('skipped', False) and "scaled_cpu_mean" in v_res:
                    final_cpu_mean = v_res["scaled_cpu_mean"]
                    final_cpu_std = v_res["scaled_cpu_std"]
                    break

            item = {
                "size": size,
                "mean_cpu": final_cpu_mean,
                "std_cpu": final_cpu_std,
                "variants": variant_results
            }
            self.results.append(item)

        self.print_summary_table()
        self.export_results(results_dir)
        self.plot_results(results_dir)

    def print_summary_table(self):
        """Affiche un tableau ASCII formaté des résultats pour toutes les variantes."""
        print("\n" + "-" * 100)
        print(f"--- RÉSULTATS SYNTHÉTIQUES : {self.name} ---")
        print("-" * 100)
        
        header = f"{'Taille N':<15} | {'CPU (s)':<12}"
        for var in self.variants:
            var_short = var[:12]
            header += f" | {var_short:<12} | {'Speedup':<8}"
        print(header)
        print("-" * 100)
        
        for r in self.results:
            line = f"{r['size']:<15,} | {r['mean_cpu']:<12.4f}"
            for var in self.variants:
                v_res = r['variants'].get(var, {})
                if v_res.get('skipped', False):
                    line += f" | {'Skipped':<12} | {'N/A':<8}"
                else:
                    mean_val = v_res.get('mean', 0.0)
                    speedup_val = v_res.get('speedup', 0.0)
                    line += f" | {mean_val:<12.4f} | {speedup_val:<8.2f}x"
            print(line)
        print("-" * 100)

    def export_results(self, results_dir: str):
        """Exporte les résultats au format JSON."""
        reports_dir = os.path.join(results_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        json_path = os.path.join(reports_dir, f"{self.name.lower()}_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "name": self.name,
                "description": self.description,
                "num_runs": self.num_runs,
                "variants": self.variants,
                "results": self.results
            }, f, indent=2)

        print(f"\n[OK] Résultats JSON sauvegardés dans : {json_path}")

    def plot_results(self, results_dir: str):
        """
        Génère un graphique en barres groupées (CPU vs Variantes).
        Affiche les valeurs de speedup directement sous forme d'annotations textuelles
        au-dessus des barres des variantes. Pas d'axe Y secondaire.
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
        x_labels = [f"{s:,}" for s in sizes]
        x = np.arange(len(sizes))
        
        num_bars = 1 + len(self.variants)
        width = 0.8 / num_bars

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0B0F19')
        ax.set_facecolor('#0B0F19')

        colors = ['#FF0055', '#00F0FF', '#F59E0B', '#10B981', '#6366F1', '#FBBF24', '#EC4899', '#A855F7']

        # 1. Barre CPU
        cpu_times = [r['mean_cpu'] for r in self.results]
        rects_cpu = ax.bar(x - 0.4 + width/2, cpu_times, width, label='CPU (Reference)', color=colors[0], alpha=0.85)

        # Annotation CPU
        for rect in rects_cpu:
            h = rect.get_height()
            if h > 0:
                ax.text(rect.get_x() + rect.get_width()/2., h + (max(cpu_times) * 0.01),
                        f"{h:.3f}s", ha='center', va='bottom', fontsize=8, color='#CBD5E1')

        # 2. Barres des variantes
        for idx, var in enumerate(self.variants):
            var_times = []
            speedups = []
            skipped_list = []
            
            for r in self.results:
                v_res = r['variants'].get(var, {})
                if v_res.get('skipped', False):
                    var_times.append(0.0)
                    speedups.append(None)
                    skipped_list.append(True)
                else:
                    var_times.append(v_res.get('mean', 0.0))
                    speedups.append(v_res.get('speedup', 0.0))
                    skipped_list.append(False)
            
            offset_pos = x - 0.4 + width/2 + (idx + 1) * width
            color = colors[(idx + 1) % len(colors)]
            rects_var = ax.bar(offset_pos, var_times, width, label=var, color=color, alpha=0.85)

            # Annotations des speedups au-dessus des barres variants
            for bar_idx, rect in enumerate(rects_var):
                if skipped_list[bar_idx]:
                    ax.text(rect.get_x() + rect.get_width()/2., 0.01,
                            "N/A", ha='center', va='bottom', fontsize=8, color='#64748B', fontstyle='italic')
                else:
                    h = rect.get_height()
                    sp = speedups[bar_idx]
                    ax.text(rect.get_x() + rect.get_width()/2., h + (max(cpu_times) * 0.01),
                            f"{sp:.1f}x\n({h:.3f}s)", ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

        ax.set_xlabel('Taille du problème (N)', fontsize=12, color='#F8FAFC', labelpad=10)
        ax.set_ylabel('Temps d\'exécution (Secondes)', fontsize=12, color='#F8FAFC', labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, color='#CBD5E1')
        ax.tick_params(colors='#CBD5E1')
        ax.grid(True, linestyle='--', alpha=0.10, color='#94A3B8')

        metadata = self.get_plot_metadata() if hasattr(self, 'get_plot_metadata') else None
        if metadata:
            meta_text = "  |  ".join([f"{k}: {v}" for k, v in metadata.items()])
            plt.title(f"GPUsync Benchmark — {self.name}\n{meta_text}", fontsize=13, color='#F8FAFC', pad=20, fontweight='bold')
        else:
            plt.title(f"GPUsync Benchmark — {self.name}", fontsize=14, color='#F8FAFC', pad=20, fontweight='bold')
        ax.legend(loc='upper left', facecolor='#1E293B', edgecolor='#334155')

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"{self.name.lower()}_benchmark.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"[OK] Graphique PNG sauvegardé dans : {plot_path}")
