import time
import asyncio
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import os
import sys

# Permet d'importer le dossier gpusync parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpusync.scheduler import Scheduler
from knn_dtw_multi_engine import knn_dtw_multi_search

async def main():
    print("=== Démonstration du Moteur KNN-DTW Multi-Query (Shape Dictionary) ===")
    
    scheduler = Scheduler(backend="opencl")
    await scheduler.initialize()
    
    # Paramètres
    N = 1_000_000  # Taille de la série temporelle
    M = 128        # Taille des motifs
    Q = 3          # Nombre de motifs distincts à chercher (Notre dictionnaire)
    
    print(f"\n[1] Génération de la base de données temporelle (L={N:,} points)...")
    t0 = time.time()
    # Bruit de fond
    time_series = np.random.randn(N).astype(np.float32) * 0.1
    print(f"    Générée en {time.time() - t0:.2f}s")
    
    print(f"\n[2] Création de {Q} motifs de classes différentes...")
    # Motif 0 : Une cloche (Gaussienne)
    x = np.linspace(-3, 3, M)
    motif_0 = np.exp(-x**2).astype(np.float32)
    
    # Motif 1 : Une onde sinusoïdale haute fréquence
    motif_1 = np.sin(np.linspace(0, 10 * np.pi, M)).astype(np.float32)
    
    # Motif 2 : Une rampe linéaire suivie d'une chute brutale
    motif_2 = np.zeros(M, dtype=np.float32)
    motif_2[:100] = np.linspace(0, 2, 100)
    motif_2[100:] = -1
    
    queries = np.vstack([motif_0, motif_1, motif_2])
    
    # Injections
    # Motif 0 injecté aux positions : 150000, 600000
    # Motif 1 injecté aux positions : 250000, 700000
    # Motif 2 injecté aux positions : 350000, 800000
    
    injections = {
        0: [150_000, 600_000],
        1: [250_000, 700_000],
        2: [350_000, 800_000]
    }
    
    for q_idx, positions in injections.items():
        for pos in positions:
            time_series[pos : pos + M] += queries[q_idx]
            
    print("\n[3] Lancement de la recherche Multi-Query sur GPU...")
    t_start = time.perf_counter()
    
    # On demande les 2 meilleures correspondances pour chaque requête
    results = await knn_dtw_multi_search(queries, time_series, K=2, scheduler=scheduler, chunk_size=500_000)
    
    t_end = time.perf_counter()
    print(f"[3] Recherche globale terminée en {t_end - t_start:.3f}s")
    
    print("\n--- RÉSULTATS FINAUX ---")
    
    fig, axes = plt.subplots(Q, 1, figsize=(12, Q * 3))
    if Q == 1: axes = [axes]
    
    for q in range(Q):
        print(f"\n--- Requête {q} ---")
        top_k_indices, top_k_distances = results[q]
        
        expected = injections[q]
        print(f"Indices Injectés : {expected}")
        
        for k in range(len(top_k_indices)):
            idx = top_k_indices[k]
            dist = top_k_distances[k]
            print(f"Rang {k+1} : Index = {idx} | Distance = {dist:.2f}")
            
            # Vérification (tolérance de 30 points à cause de l'élasticité du DTW sur les bords plats)
            is_correct = any(abs(idx - exp) <= 30 for exp in expected)
            if is_correct:
                print(f"[OK] Motif à {idx} détecté avec succès ! (Classe {q})")
            else:
                print(f"[ERREUR] Faux positif à {idx}")
                
        # Affichage du motif de la requête
        ax = axes[q]
        ax.plot(queries[q], 'k--', linewidth=2, label=f'Requête (Classe {q})')
        
        # Affichage des matchs trouvés
        colors = ['r', 'g', 'b', 'orange', 'purple']
        for k, idx in enumerate(top_k_indices):
            match_segment = time_series[idx : idx + M]
            ax.plot(match_segment, color=colors[k % len(colors)], label=f'Match {k+1} (idx={idx})')
            
        ax.set_title(f'Résultats pour la Requête {q} (Distance DTW)')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'knn_dtw_multi_plot.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n[5] Génération du graphique visuel...")
    print(f"Graphique sauvegardé sous : {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
