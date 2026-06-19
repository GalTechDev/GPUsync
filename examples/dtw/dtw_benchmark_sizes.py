import time
import asyncio
import numpy as np
import sys
import os
import csv
import pyopencl as cl

# Permet d'importer le dossier gpusync parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer
from gpusync.compiler import gpusync_kernel

@gpusync_kernel
def batch_dtw_streaming(ctx:int, reference: list, targets: list, targets_offset: int, seq_length: int, outputs: list):
    target_idx: int = targets_offset + ctx
    target_start: int = target_idx * seq_length
    
    # Préchargement Registres L1
    ref_local = [0.0] * 30
    tgt_local = [0.0] * 30
    
    for j in range(0, seq_length):
        ref_local[j] = reference[j]
        tgt_local[j] = targets[target_start + j]
    
    prev_row = [0.0] * 30
    curr_row = [0.0] * 30
    
    d = ref_local[0] - tgt_local[0]
    curr_row[0] = d * d
    
    for j in range(1, seq_length):
        d = ref_local[0] - tgt_local[j]
        curr_row[j] = curr_row[j - 1] + d * d
        
    for j in range(0, seq_length):
        prev_row[j] = curr_row[j]
        
    for i in range(1, seq_length):
        d = ref_local[i] - tgt_local[0]
        curr_row[0] = prev_row[0] + d * d
        
        for j in range(1, seq_length):
            d = ref_local[i] - tgt_local[j]
            cost = d * d
            m1 = prev_row[j]
            m2 = curr_row[j - 1]
            m3 = prev_row[j - 1]
            
            min_val = m1
            if m2 < min_val:
                min_val = m2
            if m3 < min_val:
                min_val = m3
                
            curr_row[j] = cost + min_val
            
        for j in range(0, seq_length):
            prev_row[j] = curr_row[j]
            
    outputs[ctx] = curr_row[seq_length - 1]

async def process_chunk(chunk_size, gpu_reference, gpu_targets_chunk, scheduler, pool, SEQ_LENGTH):
    """
    Simule la récupération des résultats mais ne stocke pas la matrice
    en RAM globale pour pouvoir tester des tailles gigantesques.
    """
    gpu_outputs = await pool.get()
    await batch_dtw_streaming(gpu_reference, gpu_targets_chunk, 0, SEQ_LENGTH, gpu_outputs, global_size=(chunk_size,), scheduler=scheduler)
    
    # On récupère les résultats depuis le GPU pour simuler la charge complète du bus PCIe
    results = await gpu_outputs.to_host()
    
    # Libération du Memory Pool
    pool.put_nowait(gpu_outputs)
    return chunk_size

async def run_benchmark_size(scheduler, t, s, n, pool):
    print(f"\n==================================================")
    print(f"Lancement Benchmark : {t} Clusters x {s:,} Cibles")
    print(f"Total Opérations    : {t * s:,} DTW")
    print(f"==================================================")

    CHUNK_SIZE = 1_000_000
    NUM_CHUNKS = s // CHUNK_SIZE
    if s % CHUNK_SIZE != 0:
        NUM_CHUNKS += 1

    np.random.seed(1)
    zz0 = np.random.random((t, n)).astype(np.float32)
    
    gpu_refs = []
    for ref_id in range(t):
        gpu_refs.append(await AsyncBuffer.to_device(scheduler, zz0[ref_id].flatten()))
        
    start_time = time.time()
    active_tasks = set()
    completed_targets = 0
    total_calculations = t * s
    
    for chunk_id in range(NUM_CHUNKS):
        offset = chunk_id * CHUNK_SIZE
        current_chunk_size = min(CHUNK_SIZE, s - offset)
        
        # ASTUCE RAM : Génération à la volée du chunk !
        # Le Garbage Collector Python supprimera ce morceau de la RAM dès qu'il ne sert plus.
        # Ça nous évite d'exploser les 64 Go de RAM système pour des tests à 256 Millions+.
        chunk_data = np.random.random((current_chunk_size, n)).astype(np.float32).flatten()
        gpu_targets_chunk = await AsyncBuffer.to_device(scheduler, chunk_data)
        
        for ref_id in range(t):
            task = asyncio.create_task(process_chunk(
                current_chunk_size, gpu_refs[ref_id], gpu_targets_chunk, 
                scheduler, pool, n
            ))
            active_tasks.add(task)
            
        while len(active_tasks) >= t * 2:
            done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            for coro in done:
                res_size = coro.result()
                completed_targets += res_size
                elapsed = time.time() - start_time
                sys.stdout.write(f"\rProgress: {completed_targets:,} / {total_calculations:,} | Débit: {completed_targets/elapsed:,.0f} / sec")
                sys.stdout.flush()

    while len(active_tasks) > 0:
        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        for coro in done:
            res_size = coro.result()
            completed_targets += res_size
            elapsed = time.time() - start_time
            sys.stdout.write(f"\rProgress: {completed_targets:,} / {total_calculations:,} | Débit: {completed_targets/elapsed:,.0f} / sec")
            sys.stdout.flush()
            
    total_time = time.time() - start_time
    debit_final = total_calculations / total_time
    print(f"\n-> Benchmark terminé en {total_time:.2f} s | Débit Final: {debit_final:,.0f} / sec")
    
    return s, total_time, debit_final

async def main():
    scheduler = Scheduler()
    await scheduler.initialize()

    n = 30
    t = 100
    
    IN_FLIGHT_LIMIT = 40
    pool = asyncio.Queue(maxsize=IN_FLIGHT_LIMIT)
    dummy_out = np.zeros(1_000_000, dtype=np.float32)
    for _ in range(IN_FLIGHT_LIMIT):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_out))

    # Tailles de S (Cibles) à tester 
    # (On va jusqu'à 256 Millions pour prouver que ni la VRAM ni la RAM ne craquent)
    sizes_to_test = [
        1024 * 1024 * 1,   # ~1 Million
        1024 * 1024 * 16,  # ~16 Millions
        1024 * 1024 * 64,  # ~64 Millions
        1024 * 1024 * 128, # ~128 Millions
        1024 * 1024 * 256  # ~256 Millions
    ]
    
    results = []
    
    # Nombre d'itérations demandées par l'utilisateur pour lisser le bruit
    NUM_ITERATIONS = 20
    
    for s in sizes_to_test:
        print(f"\n--- Lancement des {NUM_ITERATIONS} tests pour la charge : {s:,} ---")
        debits_for_s = []
        for i in range(NUM_ITERATIONS):
            size, t_time, debit = await run_benchmark_size(scheduler, t, s, n, pool)
            debits_for_s.append(debit)
            
        mean_debit = float(np.mean(debits_for_s))
        std_debit = float(np.std(debits_for_s))
        
        results.append({
            "Cibles": s, 
            "Calculs_Totaux": s * t * NUM_ITERATIONS, 
            "Temps_sec": "N/A", 
            "Debit_sec_mean": mean_debit,
            "Debit_sec_std": std_debit
        })
        
    # Sauvegarde des résultats dans le dossier benchmark
    benchmark_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark", "dtw")
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)
        
    csv_path = os.path.join(benchmark_dir, "results.csv")
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Cibles", "Calculs_Totaux", "Temps_sec", "Debit_sec_mean", "Debit_sec_std"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n[SUCCES] Table de résultats sauvegardée dans : {csv_path}")

    # --- GENERATION DU GRAPHIQUE (PLOT) ---
    try:
        import matplotlib.pyplot as plt
        
        # Extraction des données
        x_labels = [f"{r['Cibles'] // 1_000_000}M" for r in results]
        y_debits = [r['Debit_sec_mean'] / 1_000_000 for r in results]
        y_errs = [r['Debit_sec_std'] / 1_000_000 for r in results]
        
        plt.figure(figsize=(10, 6))
        
        # Tracé de la courbe moyenne avec barres d'erreur
        plt.errorbar(x_labels, y_debits, yerr=y_errs, marker='o', linestyle='-', color='#00a8ff', linewidth=3, markersize=10, capsize=5, label='Moyenne GPUsync (20 Itérations)')
        
        # Esthétique du graphique pour le rapport
        plt.title("Performance GPUsync (True Streaming) : Évolution du Débit selon la Charge", fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("Taille du Dataset Streaming (Nombre d'Individus Cibles)", fontsize=12, fontweight='bold')
        plt.ylabel("Débit de Calcul (Millions de DTW / sec)", fontsize=12, fontweight='bold')
        
        # On force l'axe Y à commencer à 0 pour bien prouver que la ligne est parfaitement "plate" en haut
        plt.ylim(0, max(y_debits) * 1.2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Ajouter les valeurs sur les points
        for i, txt in enumerate(y_debits):
            plt.annotate(f"{txt:.1f} M/s", (x_labels[i], y_debits[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
        
        # Sauvegarde du graphique en haute résolution
        plot_path = os.path.join(benchmark_dir, "throughput_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCES] Graphique de performance sauvegardé dans : {plot_path}")
        
    except ImportError:
        print("\n[INFO] La librairie 'matplotlib' n'est pas installée.")
        print("-> Exécutez 'pip install matplotlib' dans votre terminal pour générer automatiquement le graphique !")

if __name__ == "__main__":
    asyncio.run(main())
