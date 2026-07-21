import asyncio
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stomp_engine import stomp_matrix_profile
from gpusync.scheduler import Scheduler

async def main():
    print("=== Démonstration du Moteur GPU-STOMP (Matrix Profile) ===")
    
    # 1. Génération des données temporelles
    # On va générer L = 100 000 points. 
    # Pour STOMP c'est O(N^2), 100k -> 10^10 itérations, ce qui prend ~0.1s sur GPU.
    # Pour bien voir la puissance du GPU, on va faire L = 50 000 points
    L = 50000
    M = 200
    
    print(f"[1] Génération de la base temporelle (L={L} points)...")
    np.random.seed(42)
    # Bruit blanc lissé (Marche aléatoire)
    T = np.cumsum(np.random.randn(L))
    
    # Injection d'un Motif (schéma récurrent)
    # Un motif est répété à deux endroits : il aura un MP très bas (vallée)
    motif_idx1 = 10000
    motif_idx2 = 35000
    motif = np.sin(np.linspace(0, 4*np.pi, M)) * 20
    T[motif_idx1:motif_idx1+M] = motif
    T[motif_idx2:motif_idx2+M] = motif
    
    # Injection d'un Discord (anomalie unique)
    # Un discord est unique : il aura un MP très haut (pic)
    discord_idx = 25000
    discord = np.exp(-np.linspace(-2, 2, M)**2) * 50
    T[discord_idx:discord_idx+M] = discord
    
    print("    Motif injecté à 10000 et 35000")
    print("    Discord injecté à 25000")
    
    scheduler = Scheduler(backend="opencl")
    await scheduler.initialize()
    
    try:
        mp = await stomp_matrix_profile(T, M, scheduler)
        
        # 3. Traitement des résultats
        mp_min = np.nanmin(mp)
        mp_max = np.nanmax(mp[mp != np.inf])
        print(f"\n[STOMP] Matrix Profile généré. Min={mp_min:.2f}, Max={mp_max:.2f}")
        
        # Trouver les indices du motif et du discord
        detected_motif_idx = np.nanargmin(mp)
        detected_discord_idx = np.nanargmax(np.where(mp == np.inf, -1, mp))
        
        print(f"    -> Meilleur Motif (Vallée) détecté à l'index: {detected_motif_idx}")
        print(f"    -> Pire Discord (Pic) détecté à l'index: {detected_discord_idx}")
        
        # 4. Graphique visuel
        print("\n[3] Génération du graphique visuel...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        ax1.plot(T, color='gray', alpha=0.7, label='Time Series')
        ax1.axvline(motif_idx1, color='green', linestyle='--', label='Motif Injected 1')
        ax1.axvline(motif_idx2, color='green', linestyle='--', label='Motif Injected 2')
        ax1.axvline(discord_idx, color='red', linestyle='--', label='Discord Injected')
        ax1.set_title('Série Temporelle avec Anomalies et Motifs')
        ax1.legend()
        
        ax2.plot(mp, color='blue', label='Matrix Profile (Distance au plus proche voisin)')
        ax2.axvline(detected_motif_idx, color='green', linestyle=':', label='Motif Détecté (Min)')
        ax2.axvline(detected_discord_idx, color='red', linestyle=':', label='Discord Détecté (Max)')
        ax2.set_title('Matrix Profile généré par GPU-STOMP')
        ax2.set_ylabel('Distance')
        ax2.legend()
        
        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stomp_plot.png')
        plt.savefig(save_path)
        print(f"Graphique sauvegardé sous : {save_path}")
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    asyncio.run(main())
