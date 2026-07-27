
import numpy as np
import pandas as pd

ref_file = "tests/ref_dtw_converted_RomaneDataSet_OXYGENE.csv"
gpu_file = "tests/dtw_matrix_oxygen.csv"

try:
    df_ref = pd.read_csv(ref_file, index_col=0)
    df_gpu = pd.read_csv(gpu_file, index_col=0)

    # Check shapes
    print(f"Shape de la référence: {df_ref.shape}")
    print(f"Shape du GPU: {df_gpu.shape}")

    # Extraire les 100 premières colonnes de la matrice GPU pour comparer
    # La matrice GPU a pour colonnes les elev_id. On suppose que ref_0 correspond au 1er elev_id, etc.
    if df_ref.shape[1] == 100:
        gpu_sub = df_gpu.iloc[:, :100].values
        ref_sub = df_ref.values

        diff = np.abs(gpu_sub - ref_sub)
        print(f"Différence maximale (MAE): {np.mean(diff)}")
        print(f"Différence max absolue: {np.max(diff)}")

        # Afficher quelques valeurs pour comparer directement
        print("\nExemple de valeurs (Ligne 0):")
        print("GPU :", gpu_sub[0, :5])
        print("REF :", ref_sub[0, :5])

        print("\nExemple de valeurs (Ligne 1):")
        print("GPU :", gpu_sub[1, :5])
        print("REF :", ref_sub[1, :5])
except Exception as e:
    print(f"Erreur: {e}")
