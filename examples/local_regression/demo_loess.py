import asyncio
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from loess_engine import loess_gpu_fit
from gpusync.scheduler import Scheduler

def generate_noisy_timeseries(N=500000):
    np.random.seed(42)
    t = np.linspace(0, 100, N)
    
    # Tendance Macro (Très Lente)
    macro_trend = 10 * np.sin(0.05 * t) + 0.5 * t
    
    # Cycles Moyen-Terme
    mid_cycles = 5 * np.sin(2.0 * t)
    
    # Bruit Haute-Fréquence
    noise = np.random.randn(N) * 3.0
    
    Y = macro_trend + mid_cycles + noise
    return t, Y, macro_trend, mid_cycles

async def main():
    print("=== Démonstration de Régression Locale (LOESS/LOWESS) sur GPU ===")
    N = 500000
    
    print(f"[1] Génération d'une série temporelle complexe (N={N} points)...")
    t, Y, macro_trend, mid_cycles = generate_noisy_timeseries(N)
    
    scheduler = Scheduler(backend="opencl")
    await scheduler.initialize()
    
    try:
        print("\n[2] Lissage Court-Terme (Débruitage) avec W = 500...")
        # W = 500 points de chaque côté
        Y_smooth_short = await loess_gpu_fit(Y, W=500, scheduler=scheduler)
        
        print("\n[3] Extraction de Tendance Macro avec W = 10000...")
        # W = 10000 points de chaque côté
        # Ce calcul prendrait des heures sur CPU ! (10 milliards d'opérations WLS)
        Y_smooth_macro = await loess_gpu_fit(Y, W=10000, scheduler=scheduler)
        
        print("\n[4] Génération du graphique visuel...")
        # Pour le graphique, on n'affiche qu'un sous-ensemble pour ne pas exploser la RAM de Matplotlib
        plot_N = min(N, 100000)
        
        plt.figure(figsize=(15, 8))
        
        # Afficher la série originale avec un faible alpha
        plt.scatter(t[:plot_N], Y[:plot_N], color='gray', s=1, alpha=0.3, label='Série Brute (Bruitée)')
        
        # Afficher les lissages
        plt.plot(t[:plot_N], Y_smooth_short[:plot_N], color='blue', linewidth=2, label='LOESS W=500 (Cycles)')
        plt.plot(t[:plot_N], Y_smooth_macro[:plot_N], color='red', linewidth=3, label='LOESS W=10000 (Tendance Macro)')
        
        # Afficher la vérité terrain pour vérifier
        plt.plot(t[:plot_N], macro_trend[:plot_N], color='black', linestyle='--', linewidth=2, label='Vraie Tendance Macro')
        
        plt.title('Lissage LOESS sur GPU : Débruitage et Extraction de Tendance')
        plt.xlabel('Temps')
        plt.ylabel('Valeur')
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'loess_plot.png')
        plt.savefig(save_path)
        print(f"Graphique sauvegardé sous : {save_path}")
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    asyncio.run(main())
