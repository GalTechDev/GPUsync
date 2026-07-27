import asyncio
import os
import sys

import numpy as np
import pandas as pd

# Ajouter le chemin racine au sys.path pour les imports gpusync
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.algorithms.dtw.dtw_gpu_engine import compute_dtw_matrix
from gpusync import Scheduler, list_gpus


async def main():
    # 1. Charger le dataset
    xlsx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RomaneDataSet_OXYGENE.xlsx")
    df = pd.read_excel(xlsx_path)
    # La colonne elev_id est l'index, le reste sont les valeurs de séries temporelles
    data_np = np.ascontiguousarray(df.iloc[:, 1:].values, dtype=np.float32)
    print(f"Shape du dataset : {data_np.shape} (N={data_np.shape[0]} séries, M={data_np.shape[1]} points)")

    # 2. Initialiser le scheduler
    gpus = list_gpus()
    if not gpus:
        raise RuntimeError("Aucun GPU compatible détecté.")
    scheduler = Scheduler(gpu=gpus[0], backend="cuda")
    await scheduler.initialize()
    print(f"Utilisation du GPU : {gpus[0].name} ({gpus[0].backend})")

    # 3. Calculer la matrice par blocs de K
    K_chunk = 16  # Taille de bloc sécuritaire pour le nombre de paramètres de kernel GPU
    N = len(data_np)
    results = []

    print("Calcul de la matrice DTW (w=5) en cours par blocs...")
    for i in range(0, N, K_chunk):
        refs_chunk = data_np[i : i + K_chunk]
        # compute_dtw_matrix retourne une matrice de forme (len(refs_chunk), N)
        dist_chunk = await compute_dtw_matrix(data_np, refs_chunk, scheduler=scheduler, w=5)
        results.append(dist_chunk)
        print(f"Progression : lignes {i} à {min(i + K_chunk, N)} / {N}")

    pairwise_dist = np.vstack(results)
    print(f"\nCalcul terminé ! Shape de la matrice finale : {pairwise_dist.shape}")

    # 4. Sauvegarder le résultat
    npy_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dtw_matrix_oxygen.npy")
    csv_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dtw_matrix_oxygen.csv")

    np.save(npy_out, pairwise_dist)
    df_dist = pd.DataFrame(pairwise_dist, index=df["elev_id"], columns=df["elev_id"])
    df_dist.to_csv(csv_out)
    print(f"Matrice de distances sauvegardée avec succès :\n - {csv_out}\n - {npy_out}")


if __name__ == "__main__":
    asyncio.run(main())
