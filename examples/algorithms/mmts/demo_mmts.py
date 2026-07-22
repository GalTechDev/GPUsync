# -*- coding: utf-8 -*-
"""
Script de démonstration pour le module MMTS (Clustering X-Means & Multi-Variables avec DTW).
Montre comment initialiser l'algorithme, exécuter le clustering sur CPU/GPU
et structurer les résultats sous forme de DataFrame.
"""

import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.algorithms.mmts.mmts_engine import XmeansTS, MultiXmeansTS
from gpusync import Scheduler, list_gpus
from config import ID_COLUMN

def generate_sample_data(n_series=100, length=32):
    """
    Génère un jeu de données de test simple avec 3 motifs temporels différents.
    """
    print(f"[+] Génération de {n_series} séries de longueur {length}...")
    np.random.seed(42)
    data = []
    
    # 3 patterns différents
    for i in range(n_series):
        t = np.arange(length)
        pattern_type = i % 3
        
        if pattern_type == 0:
            # Pattern 1: Sinusoïdal simple
            sig = 15.0 + 5.0 * np.sin(t * (2 * np.pi / 16))
        elif pattern_type == 1:
            # Pattern 2: Rampe croissante
            sig = 10.0 + 10.0 * (t / length)
        else:
            # Pattern 3: Créneau / Marche
            sig = np.full(length, 12.0)
            sig[length//2:] = 22.0
            
        # Ajout d'un bruit gaussien léger
        sig += np.random.randn(length) * 0.5
        data.append(sig.astype(np.float32))
        
    df_ids = pd.DataFrame({
        ID_COLUMN: [f"TS_{i:04d}" for i in range(n_series)]
    })
    
    return data, df_ids

def main():
    print("=" * 80)
    print("                 DÉMO MMTS (Clustering X-Means DTW)")
    print("=" * 80)
    
    # 1. Génération des données de démo
    n_series = 60
    length = 32
    data, df_ids = generate_sample_data(n_series, length)
    
    # Paramètres de l'algorithme
    k_clusters = 3
    w_window = 5
    nb_instance_min = 5
    
    # 2. Initialisation de GPUsync (si disponible)
    gpus = list_gpus()
    scheduler = None
    use_gpu = False
    
    if not gpus:
        print("[!] Erreur : GPUsync requiert un GPU compatible OpenCL pour fonctionner. Fin de la démo.")
        return
        
    print(f"[+] GPU détecté : {gpus[0]}")
    scheduler = Scheduler(gpu=gpus[0])
    scheduler.initialize()
    print("[+] GPUsync initialisé avec succès.")

    # 3. Exécution avec le moteur optimisé GPU
    print(f"\n[+] Démarrage du clustering X-Means avec DTW (K={k_clusters}, W={w_window})...")
    
    # Création du moteur GPU
    engine = XmeansTS(
        data=data,
        numclust=k_clusters,
        file="demo_run",
        listID=df_ids,
        coeffRap=1.0,
        scheduler=scheduler
    )
    
    t0 = time.perf_counter()
    # Lancement du pipeline X-means récursif
    # Note : Le dictionnaire distCompare et la liste listeIDOk1 sont requis par l'API historique
    dist_compare = {}
    liste_ok = []
    
    results_df = engine.k_means_clust(
        data=data,
        num_clust=k_clusters,
        num_iter=1,
        listDataFrame=df_ids.copy(),
        LastnbClust=0,
        tabEnviIndice="Set_TEMP",
        distCompare=dist_compare,
        listeIDOk1=liste_ok,
        nbInstanceMinim=nb_instance_min,
        w=w_window
    )
    t1 = time.perf_counter()
    
    print(f"[+] Clustering terminé en {(t1-t0)*1000:.2f} ms.")
    
    # 4. Affichage des résultats
    if results_df is not None:
        print("\n[+] Matrice de distances finale récupérée avec succès.")
        print(f"    Dimensions : {results_df.shape}")
        print("    Aperçu des premières lignes :")
        print("-" * 75)
        print(results_df.head(10).to_string(index=False))
        print("-" * 75)
        
        # Reconstruction des assignations de clusters à partir de l'attribut assignments
        print("\n[+] Répartition par cluster (depuis engine.assignments) :")
        print("-" * 60)
        total_assigned = 0
        for clust, members in engine.assignments.items():
            print(f"    Cluster {clust} : {len(members):>3d} séries (ex: {members[:4]})")
            total_assigned += len(members)
        print(f"    Total assignés : {total_assigned} séries sur {n_series}")
        print("-" * 60)
    else:
        print("[!] Le clustering a retourné des données vides.")
        
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
