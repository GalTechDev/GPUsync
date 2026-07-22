import os
import sys
import time
import numpy as np
import pandas as pd

# Ajout du chemin du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpusync import AsyncBuffer, Scheduler, list_gpus

def is_git_lfs_pointer(file_path):
    if not os.path.exists(file_path):
        return True
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            return "version https://git-lfs.github.com" in first_line
    except Exception:
        return False

def generate_mock_dataset(xlsx_path, csv_path):
    # Génère un signal temporel sinusoïdal réaliste (ex: oscillation thermique diurne)
    np.random.seed(42)
    n_points = 5000
    timestamps = pd.date_range(start="2026-07-22 00:00:00", periods=n_points, freq="15min")
    
    # Signal oscillant avec du bruit
    values = 20.0 + 5.0 * np.sin(np.arange(n_points) * (2 * np.pi / 96)) + np.random.randn(n_points) * 0.5
    df = pd.DataFrame({
        "timestamp": timestamps,
        "value": values
    })
    
    # Sauvegarde en CSV
    df.to_csv(csv_path, index=False)
    print(f"    [OK] Fichier CSV factice écrit dans : {csv_path}")
    
    # Essai d'écriture en XLSX (si openpyxl est disponible)
    try:
        df.to_excel(xlsx_path, index=False)
        print(f"    [OK] Fichier XLSX factice écrit dans : {xlsx_path}")
    except Exception as e:
        print(f"    [!] Écriture du XLSX factice ignorée (openpyxl non installé ou erreur) : {e}")

def convert_xlsx_to_csv(xlsx_path, csv_path):
    if is_git_lfs_pointer(xlsx_path):
        print(f"[!] {xlsx_path} est un pointeur Git LFS (données réelles non chargées).")
        generate_mock_dataset(xlsx_path, csv_path)
    else:
        print(f"[+] Conversion de {xlsx_path} vers {csv_path}...")
        try:
            df = pd.read_excel(xlsx_path)
            df.to_csv(csv_path, index=False)
            print(f"[OK] Conversion réussie : {csv_path}")
        except Exception as e:
            print(f"[X] Erreur lors de la conversion de {xlsx_path} : {e}")
            print("    -> Génération de données factices en repli...")
            generate_mock_dataset(xlsx_path, csv_path)

def main():
    print("=====================================================================")
    # 1. Sélectionner et convertir un fichier de données de RomaneDataSet
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    xlsx_file = "RomaneDataSet_TEMP.xlsx"
    csv_file = "RomaneDataSet_TEMP.csv"
    
    xlsx_path = os.path.join(data_dir, xlsx_file)
    csv_path = os.path.join(data_dir, csv_file)
    
    # Conversion automatique XLSX -> CSV avant traitement
    convert_xlsx_to_csv(xlsx_path, csv_path)
    
    # 2. Chargement du fichier CSV généré
    if not os.path.exists(csv_path):
        print(f"[X] Impossible de charger le fichier CSV final à {csv_path}.")
        return

    df = pd.read_csv(csv_path)
    # Utilise la colonne des valeurs (value pour mock, ou la colonne correspondante)
    val_col = "value" if "value" in df.columns else df.columns[1]
    data_values = df[val_col].values.astype(np.float32)
    
    print(f"[+] Chargement réussi : {len(data_values):,} points de données de température.")
    
    # 3. Segmentation en fenêtres glissantes de longueur n = 32
    n = 32
    num_windows = len(data_values) - n + 1
    if num_windows <= 0:
        print("[X] Le jeu de données est trop court pour la segmentation.")
        return
        
    targets = np.zeros((num_windows, n), dtype=np.float32)
    for i in range(num_windows):
        targets[i] = data_values[i : i + n]
        
    # 4. Création d'une séquence de référence (par ex: une oscillation thermique diurne parfaite)
    ref = 20.0 + 5.0 * np.sin(np.arange(n) * (2 * np.pi / 32))
    ref = ref.astype(np.float32)
    
    # 5. Exécution du DTW accéléré sur GPU
    gpus = list_gpus()
    if not gpus:
        print("[X] Aucun GPU compatible GPUsync trouvé.")
        return
        
    sched = Scheduler(gpu=gpus[0])
    sched.initialize()
    
    from examples.algorithms.dtw.dtw_gpu_engine import get_batch_dtw_kernel
    kernel = get_batch_dtw_kernel(32, 1)
    
    # Remplissage et envoi en VRAM
    buf_ref = AsyncBuffer.to_device(sched, ref)
    buf_tgt = AsyncBuffer.to_device(sched, targets.flatten())
    buf_out = AsyncBuffer.empty_like(sched, np.zeros(num_windows, dtype=np.float32))
    
    print("[+] Lancement du calcul DTW 1-to-N sur GPU en parallèle...")
    t0 = time.perf_counter()
    kernel(
        buf_ref, buf_tgt, 0, 32, buf_out,
        1, num_windows, global_size=(num_windows,), scheduler=sched
    )
    
    distances = buf_out.to_host()
    gpu_time = time.perf_counter() - t0
    
    # 6. Affichage des résultats
    top_indices = np.argsort(distances)[:5]
    print("\n" + "=" * 60)
    print("--- RÉSULTATS DU PIPELINE DTW APPLIQUÉ (GPUsync) ---")
    print(f"Fichier analysé                    : {xlsx_file} (converti en CSV)")
    print(f"Nombre total de fenêtres analysées : {num_windows:,}")
    print(f"Temps GPU de calcul DTW            : {gpu_time * 1000:.2f} ms")
    print("=" * 60)
    
    for rank, idx in enumerate(top_indices):
        print(f"Top {rank + 1}: Fenêtre à l'index {idx:<5} | Distance DTW minimale: {distances[idx]:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
